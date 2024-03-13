use std::{
    collections::HashSet,
    env::current_exe,
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

// TODO: Make this 2047 or something so that mipmapping works.
const GRID_SIZE: u32 = 128;
const TRACE_SIZE: u32 = 256;
const TRACE_LENGTH: u32 = 146;
const SCALING: u32 = 16;
const NUM_DIRECTIONS: u32 = 64;
const BLUR: f32 = 0.3;

#[derive(Debug, Clone, Copy)]
struct Runtime {
    cursor_pos: PhysicalPosition<f64>,
    t: u32,
    dir: u32,
}
impl Default for Runtime {
    fn default() -> Self {
        Self {
            cursor_pos: PhysicalPosition::new(0.0, 0.0),
            t: 0,
            dir: 0,
        }
    }
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
                color * Vec3::expr(0.5, 0.0, 0.0)
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
        &track!(|dir| {
            set_block_size([TRACE_SIZE, 1, 1]);
            // let dir = block_id().y;
            let index = dispatch_id().x;
            let angle = (dir.cast_f32() * 2.0 * std::f32::consts::PI) / NUM_DIRECTIONS as f32;
            let quadrant = ((dir + NUM_DIRECTIONS / 8) / (NUM_DIRECTIONS / 4)) % 4;
            let quadrant_angle = ((dir.cast_f32() / NUM_DIRECTIONS as f32)
                - (quadrant.cast_f32() / 4.0))
                * 2.0
                * std::f32::consts::PI;
            let slope = quadrant_angle.tan();
            let correction = (1.0 + slope * slope).sqrt();
            let offset_slope = quadrant_angle.sin() / correction;
            let primary_axis = if quadrant == 0 {
                Vec2::<i32>::expr(1, 0)
            } else if quadrant == 1 {
                Vec2::<i32>::expr(0, 1)
            } else if quadrant == 2 {
                Vec2::<i32>::expr(-1, 0)
            } else {
                Vec2::<i32>::expr(0, -1)
            };
            let secondary_axis = Vec2::<i32>::expr(-primary_axis.y, primary_axis.x);
            let light = 0.0_f32.var();
            let block_offset = Vec2::<f32>::splat(64.0)
                - (TRACE_LENGTH as f32 / 2.0) * Vec2::expr(angle.cos(), angle.sin()) * correction
                - (TRACE_SIZE as f32 / 2.0) * Vec2::expr(-angle.sin(), angle.cos()) / correction;
            let block_offset = block_offset.cast_i32();
            let offset = block_offset + index.cast_i32() * secondary_axis;
            let shared = Shared::<f32>::new(TRACE_SIZE as usize + 2);

            for i in 0..TRACE_LENGTH {
                let i = i.cast_i32() - (index.cast_f32() * offset_slope).floor().cast_i32();

                let si = index + 1;
                shared.write(si, light);
                sync_block();
                *light =
                    (1.0 - 2.0 * BLUR) * light + BLUR * (shared.read(si - 1) + shared.read(si + 1));

                let pos = offset
                    + primary_axis * i
                    + secondary_axis * (i.cast_f32() * slope).floor().cast_i32();
                if pos.x < 0 || pos.x >= GRID_SIZE as i32 || pos.y < 0 || pos.y >= GRID_SIZE as i32
                {
                    continue;
                }

                let pos = pos.cast_u32();
                let wall = walls.read(pos);
                if wall > 0.0 {
                    *light = wall;
                }
                lights.write(pos.extend(dir), light);
            }
        }),
    );

    let write_wall_kernel = Kernel::<fn(Vec2<u32>, f32)>::new(
        &device,
        &track!(|pos: Expr<Vec2<u32>>, value: Expr<f32>| {
            walls.write(pos, value);
        }),
    );

    let mut active_buttons = HashSet::new();

    let mut update_cursor = |active_buttons: &HashSet<MouseButton>, rt: &mut Runtime| {
        let pos = Vec2::new(
            (rt.cursor_pos.x as u32) / SCALING,
            (rt.cursor_pos.y as u32) / SCALING,
        );
        if active_buttons.contains(&MouseButton::Left) {
            write_wall_kernel.dispatch([1, 1, 1], &pos, &1.0);
        }
        if active_buttons.contains(&MouseButton::Right) {
            write_wall_kernel.dispatch([1, 1, 1], &pos, &0.001);
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
            _ => (),
        }
    };
    let update_keyboard = &mut update_keyboard;

    let mut rt = Runtime::default();

    let start = Instant::now();

    let dt = Duration::from_secs_f64(1.0 / 60.0);

    let mut avg_iter_time = 0.0;

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

                    if dt * rt.t < start.elapsed() {
                        let iter_st = Instant::now();
                        rt.t += 1;
                        update_cursor(&active_buttons, &mut rt);
                        {
                            let mut commands = vec![];

                            commands.extend([
                                clear_kernel.dispatch_async([GRID_SIZE, GRID_SIZE, NUM_DIRECTIONS]),
                                update_kernel.dispatch_async([TRACE_SIZE, 1, 1], &rt.dir),
                                draw_kernel.dispatch_async([
                                    GRID_SIZE * SCALING,
                                    GRID_SIZE * SCALING,
                                    1,
                                ]),
                            ]);
                            scope.submit(commands);
                        }
                        avg_iter_time = avg_iter_time * 0.9 + iter_st.elapsed().as_secs_f64() * 0.1;
                        if rt.t % 60 == 0 {
                            println!("Avg iter time: {}", avg_iter_time * 1000.0);
                        }
                    }
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
