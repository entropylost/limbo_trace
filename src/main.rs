use std::{
    collections::HashSet,
    env::current_exe,
    f32::consts::{PI, TAU},
    time::{Duration, Instant},
};

use luisa::{
    lang::{
        functions::{block_id, sync_block},
        types::{
            shared::Shared,
            vector::{Vec2, Vec3, Vec4},
        },
    },
    prelude::*,
};
use luisa_compute as luisa;
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
};

const GRID_SIZE: u32 = 128;
const TRACE_SIZE: u32 = GRID_SIZE;
const TRACE_LENGTH: u32 = GRID_SIZE;
const SCALING: u32 = 16;
const NUM_DIRECTIONS: u32 = 64;
const BLUR: f32 = 0.3;

#[derive(Debug, Clone, Copy)]
struct Runtime {
    cursor_pos: PhysicalPosition<f64>,
    t: u32,
    dir: u32,
    offset: i32,
}
impl Default for Runtime {
    fn default() -> Self {
        Self {
            cursor_pos: PhysicalPosition::new(0.0, 0.0),
            t: 0,
            dir: 0,
            offset: 0,
        }
    }
}

#[tracked]
fn sign(x: Expr<Vec2<f32>>) -> Expr<Vec2<f32>> {
    1.0.copysign(x)
}

#[tracked]
fn hash(x: Expr<u32>) -> Expr<u32> {
    let x = x.var();
    *x ^= x >> 17;
    *x *= 0xed5ad4bb;
    *x ^= x >> 11;
    *x *= 0xac4c1b51;
    *x ^= x >> 15;
    *x *= 0x31848bab;
    *x ^= x >> 14;
    **x
}

#[tracked]
fn rand(pos: Expr<Vec2<u32>>, t: Expr<u32>, c: u32) -> Expr<u32> {
    let input = t + pos.x * GRID_SIZE + pos.y * GRID_SIZE * GRID_SIZE + c * 1063; //* GRID_SIZE * GRID_SIZE * GRID_SIZE;
    hash(input)
}

#[tracked]
fn rand_f32(pos: Expr<Vec2<u32>>, t: Expr<u32>, c: u32) -> Expr<f32> {
    rand(pos, t, c).as_f32() / u32::MAX as f32
}

fn main() {
    let _ = color_eyre::install();
    luisa::init_logger();
    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_device("cuda");

    let event_loop = EventLoop::new().unwrap();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(
            GRID_SIZE * SCALING,
            GRID_SIZE * SCALING,
        ))
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    let swapchain = device.create_swapchain(
        &window,
        &device.default_stream(),
        GRID_SIZE * SCALING,
        GRID_SIZE * SCALING,
        false,
        false,
        3,
    );
    let display = device.create_tex2d::<Vec4<f32>>(
        swapchain.pixel_storage(),
        GRID_SIZE * SCALING,
        GRID_SIZE * SCALING,
        1,
    );

    let walls = device.create_tex2d::<f32>(PixelStorage::Float1, GRID_SIZE, GRID_SIZE, 1);

    let lights = device.create_tex3d::<f32>(
        PixelStorage::Float1,
        GRID_SIZE,
        GRID_SIZE,
        NUM_DIRECTIONS,
        1,
    );

    let draw_kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            let display_pos = dispatch_id().xy();
            let pos = display_pos / SCALING;
            let color: Expr<Vec3<f32>> = if walls.read(pos) > 0.0 {
                Vec3::expr(1.0, 1.0, 1.0) * (walls.read(pos) * 0.5 + 0.5)
            } else {
                let color = 0.0_f32.var();
                for i in 0..NUM_DIRECTIONS {
                    *color += lights.read(pos.extend(i));
                }
                color * Vec3::expr(1.0, 0.0, 0.0)
            };
            display.write(display_pos, color.extend(1.0));
        }),
    );

    let clear_kernel = Kernel::<fn()>::new(
        &device,
        &track!(|| {
            lights.write(dispatch_id(), 0.0);
        }),
    );

    let update_kernel = Kernel::<fn(u32)>::new(
        &device,
        &track!(|t| {
            // If block size is too big,
            // can manually run multiple rays per block.
            set_block_size([TRACE_SIZE, 1, 1]);
            let dir = block_id().y;
            let index = dispatch_id().x;
            let angle = (dir.cast_f32() * TAU) / NUM_DIRECTIONS as f32 + 0.0001;
            let quadrant = (dir / (NUM_DIRECTIONS / 4)) % 4;

            let light = 0.0_f32.var();

            let ray_dir = Vec2::expr(angle.cos(), angle.sin());
            let delta_dist = 1.0 / ray_dir.abs();
            let step = sign(ray_dir).cast_i32();

            let correction = ray_dir.x.abs() + ray_dir.y.abs();

            let trace_length = correction * correction * TRACE_LENGTH as f32;

            let ray_pos = Vec2::<f32>::splat(GRID_SIZE as f32 / 2.0)
                - (trace_length / 2.0) * Vec2::expr(angle.cos(), angle.sin()) / correction
                - (TRACE_SIZE as f32 / 2.0) * Vec2::expr(-angle.sin(), angle.cos()) * correction
                + Vec2::expr(
                    rand_f32(Vec2::expr(dir, t), 0.expr(), 0),
                    rand_f32(Vec2::expr(dir, t), 1.expr(), 0),
                )
                + index.cast_f32() * Vec2::expr(-step.y.as_f32(), step.x.as_f32())
                + index.cast_f32()
                    * 2.0_f32.sqrt()
                    * (quadrant.as_f32() * PI / 2.0 + PI / 4.0 - angle).sin()
                    * ray_dir;
            let pos = ray_pos.floor().cast_i32().var();

            let side_dist =
                (sign(ray_dir) * (pos.cast_f32() - ray_pos) + sign(ray_dir) * 0.5 + 0.5)
                    * delta_dist;
            let side_dist = side_dist.var();

            // Remove to make the light look manhattan.
            let blur = BLUR / correction;

            let shared = Shared::<f32>::new(TRACE_SIZE as usize + 2);

            for _i in 0.expr()..trace_length.cast_u32() {
                let si = index + 1;
                shared.write(si, light);
                sync_block();
                *light =
                    (1.0 - 2.0 * blur) * light + blur * (shared.read(si - 1) + shared.read(si + 1));

                let mask = side_dist <= side_dist.yx();
                *side_dist += mask.select(delta_dist, Vec2::splat_expr(0.0));
                *pos += mask.select(step, Vec2::splat_expr(0));

                if pos.x < 0 || pos.x >= GRID_SIZE as i32 || pos.y < 0 || pos.y >= GRID_SIZE as i32
                {
                    continue;
                }

                let pos = pos.cast_u32();
                lights.write(pos.extend(dir), light);
                let wall = walls.read(pos);
                if wall > 0.0 {
                    *light = wall / NUM_DIRECTIONS as f32;
                }
            }
        }),
    );

    let write_wall_kernel = Kernel::<fn(Vec2<u32>, f32)>::new(
        &device,
        &track!(|pos, value| {
            walls.write(pos + dispatch_id().xy(), value);
        }),
    );

    let mut active_buttons = HashSet::new();

    let mut update_cursor = |active_buttons: &HashSet<MouseButton>, rt: &mut Runtime| {
        let pos = Vec2::new(
            (rt.cursor_pos.x as u32) / SCALING,
            (rt.cursor_pos.y as u32) / SCALING,
        );
        if active_buttons.contains(&MouseButton::Left) {
            write_wall_kernel.dispatch([4, 4, 1], &pos, &1.0);
        }
        if active_buttons.contains(&MouseButton::Right) {
            write_wall_kernel.dispatch([4, 4, 1], &pos, &0.001);
        }
    };
    let update_cursor = &mut update_cursor;

    let mut update_keyboard = |ev: KeyEvent, rt: &mut Runtime| {
        if ev.state != ElementState::Pressed {
            return;
        }
        let PhysicalKey::Code(key) = ev.physical_key else {
            panic!("Invalid")
        };
        match key {
            KeyCode::KeyD => {
                rt.dir = (rt.dir + 1) % NUM_DIRECTIONS;
            }
            KeyCode::KeyA => {
                rt.dir = (rt.dir + NUM_DIRECTIONS - 1) % NUM_DIRECTIONS;
            }
            KeyCode::KeyW => {
                rt.offset += 1;
            }
            KeyCode::KeyS => {
                rt.offset -= 1;
            }
            _ => (),
        }
    };
    let update_keyboard = &mut update_keyboard;

    let mut rt = Runtime::default();

    let start = Instant::now();

    let dt = Duration::from_secs_f64(1.0 / 60.0);

    let mut avg_iter_time = 0.0;

    // write_wall_kernel.dispatch(
    //     [GRID_SIZE / 2, GRID_SIZE / 2, 1],
    //     &Vec2::splat(GRID_SIZE / 4),
    //     &1.0,
    // );

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run(move |event, elwt| match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => {
                    elwt.exit();
                }
                WindowEvent::RedrawRequested => {
                    let scope = device.default_stream().scope();
                    scope.present(&swapchain, &display);

                    // if dt * rt.t < start.elapsed() {
                    let iter_st = Instant::now();
                    rt.t += 1;
                    update_cursor(&active_buttons, &mut rt);
                    {
                        let mut commands = vec![];

                        for _i in 0..300 {
                            commands.push(
                                update_kernel
                                    .dispatch_async([TRACE_SIZE, NUM_DIRECTIONS, 1], &rt.t),
                            );
                        }

                        commands.extend([
                            update_kernel.dispatch_async([TRACE_SIZE, NUM_DIRECTIONS, 1], &rt.t),
                            draw_kernel.dispatch_async([
                                GRID_SIZE * SCALING,
                                GRID_SIZE * SCALING,
                                1,
                            ]),
                        ]);
                        scope.submit(commands);
                    }
                    avg_iter_time = avg_iter_time * 0.99 + iter_st.elapsed().as_secs_f64() * 0.01;
                    if rt.t % 60 == 0 {
                        println!("Avg iter time: {}", avg_iter_time * 1000.0);
                        println!("FPS: {}", rt.t as f64 / start.elapsed().as_secs_f64());
                        println!(
                            "Avg iter (ms): {}",
                            start.elapsed().as_secs_f64() / rt.t as f64 * 1000.0
                        );
                    }
                    // }
                    window.request_redraw();
                }
                WindowEvent::CursorMoved { position, .. } => {
                    rt.cursor_pos = position;
                    update_cursor(&active_buttons, &mut rt);
                }
                WindowEvent::MouseInput { button, state, .. } => {
                    match state {
                        ElementState::Pressed => {
                            active_buttons.insert(button);
                        }
                        ElementState::Released => {
                            active_buttons.remove(&button);
                        }
                    }
                    update_cursor(&active_buttons, &mut rt);
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    update_keyboard(event, &mut rt);
                }
                _ => (),
            },
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => (),
        })
        .unwrap();
}
