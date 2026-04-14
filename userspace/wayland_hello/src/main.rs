#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

use abi::ids::HandleId;
use abi::syscall::vfs_flags::O_RDWR;
use alloc::vec::Vec;
use stem::info;
use stem::syscall::{sleep_ms, vfs_open, vfs_read, vfs_write};
use stem::thing::{sys as thingsys, ThingId};

const REGISTRY_ID: u32 = 2;
const COMPOSITOR_ID: u32 = 3;
const SHM_ID: u32 = 4;
const WM_BASE_ID: u32 = 5;

const TOP_SURFACE_ID: u32 = 10;
const TOP_XDG_SURFACE_ID: u32 = 11;
const TOPLEVEL_ID: u32 = 12;

const POPUP_SURFACE_ID: u32 = 20;
const POPUP_XDG_SURFACE_ID: u32 = 21;
const POSITIONER_ID: u32 = 22;
const POPUP_ID: u32 = 23;

#[derive(Clone, Copy)]
struct BufferState {
    pool_id: u32,
    buffer_id: u32,
    bs_id: ThingId,
    ptr: *mut u8,
    width: u32,
    height: u32,
    stride: u32,
}

struct PendingSurface {
    serial: Option<u32>,
    width: u32,
    height: u32,
    dirty: bool,
}

#[stem::main]
fn main(_arg: usize) -> ! {
    let fd = connect_wayland();

    send_get_registry(fd, REGISTRY_ID);
    read_initial_globals(fd);

    bind_global(fd, 1, "wl_compositor", 4, COMPOSITOR_ID);
    bind_global(fd, 2, "wl_shm", 1, SHM_ID);
    bind_global(fd, 3, "xdg_wm_base", 1, WM_BASE_ID);

    create_surface(fd, COMPOSITOR_ID, TOP_SURFACE_ID);
    get_xdg_surface(fd, WM_BASE_ID, TOP_XDG_SURFACE_ID, TOP_SURFACE_ID);
    get_toplevel(fd, TOP_XDG_SURFACE_ID, TOPLEVEL_ID);
    set_toplevel_title(fd, TOPLEVEL_ID, "Thing-OS XDG Demo");
    set_toplevel_app_id(fd, TOPLEVEL_ID, "thingos.wayland_hello");
    commit_surface(fd, TOP_SURFACE_ID);

    let mut top_pending = PendingSurface {
        serial: None,
        width: 480,
        height: 320,
        dirty: false,
    };
    let mut popup_pending = PendingSurface {
        serial: None,
        width: 160,
        height: 96,
        dirty: false,
    };
    let mut top_buffer: Option<BufferState> = None;
    let mut popup_buffer: Option<BufferState> = None;
    let mut popup_created = false;

    loop {
        let mut in_buf = [0u8; 4096];
        let len = match vfs_read(fd, &mut in_buf) {
            Ok(n) if n > 0 => n,
            _ => {
                sleep_ms(16);
                continue;
            }
        };

        let mut offset = 0usize;
        while offset + 8 <= len {
            let (object_id, opcode, size) = decode_header(&in_buf[offset..len]);
            if size < 8 || offset + size as usize > len {
                break;
            }
            let payload = &in_buf[offset + 8..offset + size as usize];
            match (object_id, opcode) {
                (WM_BASE_ID, 0) if payload.len() >= 4 => {
                    send_pong(fd, WM_BASE_ID, read_u32(payload, 0));
                }
                (TOP_XDG_SURFACE_ID, 0) if payload.len() >= 4 => {
                    top_pending.serial = Some(read_u32(payload, 0));
                    top_pending.dirty = true;
                }
                (TOPLEVEL_ID, 0) if payload.len() >= 12 => {
                    let width = read_i32(payload, 0);
                    let height = read_i32(payload, 4);
                    if width > 0 {
                        top_pending.width = width as u32;
                    }
                    if height > 0 {
                        top_pending.height = height as u32;
                    }
                }
                (TOPLEVEL_ID, 1) => {
                    info!("wayland_hello: compositor requested close; idling");
                    idle_forever();
                }
                (POPUP_XDG_SURFACE_ID, 0) if payload.len() >= 4 => {
                    popup_pending.serial = Some(read_u32(payload, 0));
                    popup_pending.dirty = true;
                }
                (POPUP_ID, 0) if payload.len() >= 16 => {
                    let width = read_i32(payload, 8);
                    let height = read_i32(payload, 12);
                    if width > 0 {
                        popup_pending.width = width as u32;
                    }
                    if height > 0 {
                        popup_pending.height = height as u32;
                    }
                }
                (POPUP_ID, 1) => {
                    popup_created = false;
                    popup_pending.serial = None;
                }
                _ => {}
            }
            offset += size as usize;
        }

        if top_pending.dirty {
            let title = "THING-OS XDG";
            let buffer = ensure_buffer(
                fd,
                SHM_ID,
                &mut top_buffer,
                TOP_SURFACE_ID + 100,
                top_pending.width,
                top_pending.height,
            );
            render_window(buffer, title);
            ack_configure(fd, TOP_XDG_SURFACE_ID, top_pending.serial.unwrap_or(0));
            attach_buffer(fd, TOP_SURFACE_ID, buffer.buffer_id);
            commit_surface(fd, TOP_SURFACE_ID);
            top_pending.dirty = false;

            if !popup_created {
                popup_created = true;
                create_surface(fd, COMPOSITOR_ID, POPUP_SURFACE_ID);
                get_xdg_surface(fd, WM_BASE_ID, POPUP_XDG_SURFACE_ID, POPUP_SURFACE_ID);
                create_positioner(fd, WM_BASE_ID, POSITIONER_ID);
                positioner_set_size(fd, POSITIONER_ID, 160, 96);
                positioner_set_anchor_rect(fd, POSITIONER_ID, 24, 24, 100, 24);
                positioner_set_offset(fd, POSITIONER_ID, 0, 6);
                get_popup(
                    fd,
                    TOP_XDG_SURFACE_ID,
                    POPUP_XDG_SURFACE_ID,
                    POPUP_ID,
                    POSITIONER_ID,
                );
                commit_surface(fd, POPUP_SURFACE_ID);
            }
        }

        if popup_created && popup_pending.dirty {
            let buffer = ensure_buffer(
                fd,
                SHM_ID,
                &mut popup_buffer,
                POPUP_SURFACE_ID + 100,
                popup_pending.width,
                popup_pending.height,
            );
            render_popup(buffer, "POPUP");
            ack_configure(fd, POPUP_XDG_SURFACE_ID, popup_pending.serial.unwrap_or(0));
            attach_buffer(fd, POPUP_SURFACE_ID, buffer.buffer_id);
            commit_surface(fd, POPUP_SURFACE_ID);
            popup_pending.dirty = false;
        }
    }
}

fn connect_wayland() -> u32 {
    if let Err(e) = stem::fs::wait_until_exists("/run/wayland-0") {
        stem::error!("wayland_hello: failed to wait for /run/wayland-0: {:?}", e);
    }

    match vfs_open("/run/wayland-0", O_RDWR) {
        Ok(fd) => {
            info!("wayland_hello: connected to /run/wayland-0");
            fd
        }
        Err(e) => {
            stem::error!("wayland_hello: failed to open /run/wayland-0: {:?}", e);
            loop {
                sleep_ms(1000);
            }
        }
    }
}

fn read_initial_globals(fd: u32) {
    let mut buf = [0u8; 512];
    let _ = vfs_read(fd, &mut buf);
}

fn ensure_buffer(
    fd: u32,
    shm_id: u32,
    current: &mut Option<BufferState>,
    base_id: u32,
    width: u32,
    height: u32,
) -> BufferState {
    if let Some(buf) = current {
        if buf.width == width && buf.height == height {
            return *buf;
        }
    }

    let stride = width * 4;
    let size = stride * height;
    let fd_buf = thingsys::memfd_create("wl.buffer", size as usize).expect("create memfd");

    use abi::vm::{VmBacking, VmMapReq, VmProt};
    let req = VmMapReq {
        addr_hint: 0,
        len: size as usize,
        prot: VmProt::READ | VmProt::WRITE | VmProt::USER,
        flags: abi::vm::VmMapFlags::empty(),
        backing: VmBacking::File {
            fd: fd_buf,
            offset: 0,
        },
    };
    let resp = thingsys::vm_map(&req).expect("map memfd");
    let ptr = resp.addr as *mut u8;

    let pool_id = base_id;
    let buffer_id = base_id + 1;
    create_pool(fd, shm_id, pool_id, fd_buf, size);
    create_buffer(fd, pool_id, buffer_id, width, height, stride);

    let out = BufferState {
        pool_id,
        buffer_id,
        bs_id: ThingId::from_u64(fd_buf as u64),
        ptr,
        width,
        height,
        stride,
    };
    *current = Some(out);
    out
}

fn render_window(buffer: BufferState, title: &str) {
    unsafe {
        let pixels = core::slice::from_raw_parts_mut(
            buffer.ptr as *mut u32,
            (buffer.width * buffer.height) as usize,
        );
        for y in 0..buffer.height as usize {
            for x in 0..buffer.width as usize {
                let color = if y < 40 { 0xFF3A4452 } else { 0xFF14181E };
                pixels[y * buffer.width as usize + x] = color;
            }
        }
        draw_text(pixels, buffer.width as usize, 16, 12, title, 0xFFFFFFFF, 3);
        draw_text(
            pixels,
            buffer.width as usize,
            16,
            72,
            "RESIZE ME FROM THE FRAME",
            0xFF9AD1FF,
            2,
        );
        draw_text(
            pixels,
            buffer.width as usize,
            16,
            110,
            "POPUP BELOW IS XDG_POPUP",
            0xFFE7D68A,
            2,
        );
    }
}

fn render_popup(buffer: BufferState, label: &str) {
    unsafe {
        let pixels = core::slice::from_raw_parts_mut(
            buffer.ptr as *mut u32,
            (buffer.width * buffer.height) as usize,
        );
        for y in 0..buffer.height as usize {
            for x in 0..buffer.width as usize {
                let border = x < 2
                    || y < 2
                    || x + 2 >= buffer.width as usize
                    || y + 2 >= buffer.height as usize;
                pixels[y * buffer.width as usize + x] =
                    if border { 0xFFFFFFFF } else { 0xFF202830 };
            }
        }
        draw_text(pixels, buffer.width as usize, 14, 18, label, 0xFFFFFFFF, 2);
    }
}

fn draw_text(
    pixels: &mut [u32],
    stride: usize,
    x: usize,
    y: usize,
    text: &str,
    color: u32,
    scale: usize,
) {
    let mut pen_x = x;
    for ch in text.bytes() {
        draw_glyph(pixels, stride, pen_x, y, ch, color, scale);
        pen_x += 6 * scale;
    }
}

fn draw_glyph(
    pixels: &mut [u32],
    stride: usize,
    x: usize,
    y: usize,
    ch: u8,
    color: u32,
    scale: usize,
) {
    let glyph = match ch {
        b'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        b'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        b'D' => [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
        b'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        b'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        b'G' => [0x0F, 0x10, 0x10, 0x17, 0x11, 0x11, 0x0F],
        b'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        b'I' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1F],
        b'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        b'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        b'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        b'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        b'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        b'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        b'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        b'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        b'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
        b'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        b'_' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F],
        b' ' => [0; 7],
        _ => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x00, 0x08],
    };

    for (row, bits) in glyph.iter().enumerate() {
        for col in 0..5usize {
            if (bits & (1 << (4 - col))) == 0 {
                continue;
            }
            for sy in 0..scale {
                for sx in 0..scale {
                    let px = x + col * scale + sx;
                    let py = y + row * scale + sy;
                    let idx = py * stride + px;
                    if idx < pixels.len() {
                        pixels[idx] = color;
                    }
                }
            }
        }
    }
}

fn panic_forever(msg: &str) -> ! {
    stem::error!("wayland_hello: {}", msg);
    idle_forever()
}

fn idle_forever() -> ! {
    loop {
        sleep_ms(1000);
    }
}

fn send_get_registry(fd: u32, new_id: u32) {
    let mut buf = Vec::new();
    encode_header(1, 1, 12, &mut buf);
    buf.extend_from_slice(&new_id.to_ne_bytes());
    let _ = vfs_write(fd, &buf);
}

fn bind_global(fd: u32, name: u32, interface: &str, version: u32, new_id: u32) {
    let mut buf = Vec::new();
    let len = interface.len() as u32 + 1;
    let mut bytes = interface.as_bytes().to_vec();
    bytes.push(0);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    let size = 8 + 4 + 4 + bytes.len() as u16 + 4 + 4;
    encode_header(REGISTRY_ID, 0, size, &mut buf);
    buf.extend_from_slice(&name.to_ne_bytes());
    buf.extend_from_slice(&len.to_ne_bytes());
    buf.extend_from_slice(&bytes);
    buf.extend_from_slice(&version.to_ne_bytes());
    buf.extend_from_slice(&new_id.to_ne_bytes());
    let _ = vfs_write(fd, &buf);
}

fn create_surface(fd: u32, compositor_id: u32, new_id: u32) {
    let mut buf = Vec::new();
    encode_header(compositor_id, 0, 12, &mut buf);
    buf.extend_from_slice(&new_id.to_ne_bytes());
    let _ = vfs_write(fd, &buf);
}

fn get_xdg_surface(fd: u32, wm_base_id: u32, new_id: u32, surface_id: u32) {
    let mut buf = Vec::new();
    encode_header(wm_base_id, 2, 16, &mut buf);
    buf.extend_from_slice(&new_id.to_ne_bytes());
    buf.extend_from_slice(&surface_id.to_ne_bytes());
    let _ = vfs_write(fd, &buf);
}

fn get_toplevel(fd: u32, xdg_surface_id: u32, new_id: u32) {
    let mut buf = Vec::new();
    encode_header(xdg_surface_id, 1, 12, &mut buf);
    buf.extend_from_slice(&new_id.to_ne_bytes());
    let _ = vfs_write(fd, &buf);
}

fn set_toplevel_title(fd: u32, toplevel_id: u32, title: &str) {
    send_string_request(fd, toplevel_id, 2, title);
}

fn set_toplevel_app_id(fd: u32, toplevel_id: u32, app_id: &str) {
    send_string_request(fd, toplevel_id, 3, app_id);
}

fn create_positioner(fd: u32, wm_base_id: u32, new_id: u32) {
    let mut buf = Vec::new();
    encode_header(wm_base_id, 1, 12, &mut buf);
    buf.extend_from_slice(&new_id.to_ne_bytes());
    let _ = vfs_write(fd, &buf);
}

fn positioner_set_size(fd: u32, positioner_id: u32, width: i32, height: i32) {
    let mut buf = Vec::new();
    encode_header(positioner_id, 1, 16, &mut buf);
    buf.extend_from_slice(&width.to_ne_bytes());
    buf.extend_from_slice(&height.to_ne_bytes());
    let _ = vfs_write(fd, &buf);
}

fn positioner_set_anchor_rect(
    fd: u32,
    positioner_id: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
) {
    let mut buf = Vec::new();
    encode_header(positioner_id, 2, 24, &mut buf);
    buf.extend_from_slice(&x.to_ne_bytes());
    buf.extend_from_slice(&y.to_ne_bytes());
    buf.extend_from_slice(&width.to_ne_bytes());
    buf.extend_from_slice(&height.to_ne_bytes());
    let _ = vfs_write(fd, &buf);
}

fn positioner_set_offset(fd: u32, positioner_id: u32, x: i32, y: i32) {
    let mut buf = Vec::new();
    encode_header(positioner_id, 6, 16, &mut buf);
    buf.extend_from_slice(&x.to_ne_bytes());
    buf.extend_from_slice(&y.to_ne_bytes());
    let _ = vfs_write(fd, &buf);
}

fn get_popup(
    fd: u32,
    parent_xdg_surface_id: u32,
    xdg_surface_id: u32,
    popup_id: u32,
    positioner_id: u32,
) {
    let mut buf = Vec::new();
    encode_header(xdg_surface_id, 2, 20, &mut buf);
    buf.extend_from_slice(&popup_id.to_ne_bytes());
    buf.extend_from_slice(&parent_xdg_surface_id.to_ne_bytes());
    buf.extend_from_slice(&positioner_id.to_ne_bytes());
    let _ = vfs_write(fd, &buf);
}

fn ack_configure(fd: u32, xdg_surface_id: u32, serial: u32) {
    let mut buf = Vec::new();
    encode_header(xdg_surface_id, 4, 12, &mut buf);
    buf.extend_from_slice(&serial.to_ne_bytes());
    let _ = vfs_write(fd, &buf);
}

fn send_pong(fd: u32, wm_base_id: u32, serial: u32) {
    let mut buf = Vec::new();
    encode_header(wm_base_id, 3, 12, &mut buf);
    buf.extend_from_slice(&serial.to_ne_bytes());
    let _ = vfs_write(fd, &buf);
}

fn create_pool(fd: u32, shm_id: u32, pool_id: u32, bs_raw: u32, size: u32) {
    let mut buf = Vec::new();
    encode_header(shm_id, 0, 20, &mut buf);
    buf.extend_from_slice(&pool_id.to_ne_bytes());
    buf.extend_from_slice(&bs_raw.to_ne_bytes());
    buf.extend_from_slice(&size.to_ne_bytes());
    let _ = vfs_write(fd, &buf);
}

fn create_buffer(fd: u32, pool_id: u32, buffer_id: u32, width: u32, height: u32, stride: u32) {
    let mut buf = Vec::new();
    encode_header(pool_id, 0, 32, &mut buf);
    buf.extend_from_slice(&buffer_id.to_ne_bytes());
    buf.extend_from_slice(&0u32.to_ne_bytes());
    buf.extend_from_slice(&width.to_ne_bytes());
    buf.extend_from_slice(&height.to_ne_bytes());
    buf.extend_from_slice(&stride.to_ne_bytes());
    buf.extend_from_slice(&0u32.to_ne_bytes());
    let _ = vfs_write(fd, &buf);
}

fn attach_buffer(fd: u32, surface_id: u32, buffer_id: u32) {
    let mut buf = Vec::new();
    encode_header(surface_id, 1, 20, &mut buf);
    buf.extend_from_slice(&buffer_id.to_ne_bytes());
    buf.extend_from_slice(&0u32.to_ne_bytes());
    buf.extend_from_slice(&0u32.to_ne_bytes());
    let _ = vfs_write(fd, &buf);
}

fn commit_surface(fd: u32, surface_id: u32) {
    let mut buf = Vec::new();
    encode_header(surface_id, 6, 8, &mut buf);
    let _ = vfs_write(fd, &buf);
}

fn send_string_request(fd: u32, object_id: u32, opcode: u16, value: &str) {
    let mut buf = Vec::new();
    let len = value.len() as u32 + 1;
    let mut bytes = value.as_bytes().to_vec();
    bytes.push(0);
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }
    let size = 8 + 4 + bytes.len() as u16;
    encode_header(object_id, opcode, size, &mut buf);
    buf.extend_from_slice(&len.to_ne_bytes());
    buf.extend_from_slice(&bytes);
    let _ = vfs_write(fd, &buf);
}

fn encode_header(object_id: u32, opcode: u16, size: u16, buf: &mut Vec<u8>) {
    buf.extend_from_slice(&object_id.to_ne_bytes());
    buf.extend_from_slice(&(((size as u32) << 16) | opcode as u32).to_ne_bytes());
}

fn decode_header(buf: &[u8]) -> (u32, u16, u16) {
    let object_id = read_u32(buf, 0);
    let size_op = read_u32(buf, 4);
    (object_id, (size_op & 0xFFFF) as u16, (size_op >> 16) as u16)
}

fn read_u32(buf: &[u8], offset: usize) -> u32 {
    u32::from_ne_bytes(buf[offset..offset + 4].try_into().unwrap())
}

fn read_i32(buf: &[u8], offset: usize) -> i32 {
    i32::from_ne_bytes(buf[offset..offset + 4].try_into().unwrap())
}
