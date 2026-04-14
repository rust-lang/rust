#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


use abi::syscall::vfs_flags::{O_CREAT, O_RDONLY, O_TRUNC, O_WRONLY};
use abi::vm::{VmBacking, VmMapFlags, VmMapReq, VmProt};
use core::time::Duration;
use stem::info;
use stem::syscall::vfs::{vfs_close, vfs_mkdir, vfs_open, vfs_read, vfs_stat, vfs_write};
use stem::syscall::{memfd_create, vm_map};

const KIND_NET_STACK: &str = "svc.net.Stack";
const WINDOW_ID: &str = "fetchd";
const SURFACE_ID: &str = "fetchd-main";
const WIDTH: u32 = 360;
const HEIGHT: u32 = 140;
const STRIDE: u32 = WIDTH * 4;

const SESSION_ROOT: &str = "/session";
const WINDOWS_ROOT: &str = "/session/windows";
const SURFACES_ROOT: &str = "/session/surfaces";

struct BufferState {
    fd: u32,
    ptr: *mut u32,
}

fn ensure_dir(path: &str) {
    let _ = vfs_mkdir(path);
}

fn write_text(path: &str, text: &str) {
    if let Ok(fd) = vfs_open(path, O_WRONLY | O_CREAT | O_TRUNC) {
        if !text.is_empty() {
            let _ = vfs_write(fd, text.as_bytes());
        }
        let _ = vfs_close(fd);
    }
}

fn read_text(path: &str) -> alloc::string::String {
    let Ok(fd) = vfs_open(path, O_RDONLY) else {
        return alloc::string::String::new();
    };
    let size = vfs_stat(fd).map(|s| s.size as usize).unwrap_or(0);
    let mut buf = alloc::vec![0u8; size];
    if size > 0 {
        let n = vfs_read(fd, &mut buf).unwrap_or(0);
        buf.truncate(n);
    }
    let _ = vfs_close(fd);
    alloc::string::String::from_utf8_lossy(&buf).trim().into()
}

fn ensure_session_objects() {
    ensure_dir(SESSION_ROOT);
    ensure_dir(WINDOWS_ROOT);
    ensure_dir(SURFACES_ROOT);

    let window_root = alloc::format!("{}/{}", WINDOWS_ROOT, WINDOW_ID);
    let surface_root = alloc::format!("{}/{}", SURFACES_ROOT, SURFACE_ID);

    ensure_dir(&window_root);
    ensure_dir(&alloc::format!("{}/shell", window_root));
    ensure_dir(&alloc::format!("{}/shell/requested", window_root));
    ensure_dir(&alloc::format!("{}/shell/current", window_root));
    ensure_dir(&alloc::format!("{}/bind", window_root));
    ensure_dir(&alloc::format!("{}/status", window_root));

    ensure_dir(&surface_root);
    ensure_dir(&alloc::format!("{}/status", surface_root));

    write_text(&alloc::format!("{}/shell/role", window_root), "toplevel\n");
    write_text(&alloc::format!("{}/shell/title", window_root), "Network\n");
    write_text(&alloc::format!("{}/shell/app_id", window_root), "fetchd\n");
    write_text(&alloc::format!("{}/shell/current/x", window_root), "20\n");
    write_text(&alloc::format!("{}/shell/current/y", window_root), "20\n");
    write_text(
        &alloc::format!("{}/shell/current/width", window_root),
        &alloc::format!("{}\n", WIDTH),
    );
    write_text(
        &alloc::format!("{}/shell/current/height", window_root),
        &alloc::format!("{}\n", HEIGHT),
    );
    write_text(&alloc::format!("{}/shell/current/z", window_root), "10\n");
    write_text(&alloc::format!("{}/shell/current/activated", window_root), "0\n");
    write_text(&alloc::format!("{}/shell/current/maximized", window_root), "0\n");
    write_text(&alloc::format!("{}/shell/current/fullscreen", window_root), "0\n");
    write_text(&alloc::format!("{}/shell/current/resizing", window_root), "0\n");
    write_text(&alloc::format!("{}/shell/requested/maximize", window_root), "0\n");
    write_text(
        &alloc::format!("{}/shell/requested/fullscreen", window_root),
        "0\n",
    );
    write_text(&alloc::format!("{}/shell/requested/minimize", window_root), "0\n");
    write_text(
        &alloc::format!("{}/bind/surface", window_root),
        &alloc::format!("{}\n", SURFACE_ID),
    );
    write_text(&alloc::format!("{}/events", window_root), "");
    write_text(&alloc::format!("{}/status/mapped", window_root), "0\n");
    write_text(&alloc::format!("{}/status/focused", window_root), "0\n");
    write_text(
        &alloc::format!("{}/status/last_configure_serial", window_root),
        "0\n",
    );
    write_text(&alloc::format!("{}/status/client_pid", window_root), "0\n");
    write_text(&alloc::format!("{}/status/closing", window_root), "0\n");

    write_text(&alloc::format!("{}/attach", surface_root), "");
    write_text(&alloc::format!("{}/damage", surface_root), "");
    write_text(&alloc::format!("{}/commit", surface_root), "0\n");
    write_text(&alloc::format!("{}/input_region", surface_root), "");
    write_text(&alloc::format!("{}/opaque_region", surface_root), "");
    write_text(&alloc::format!("{}/status/mapped", surface_root), "0\n");
    write_text(&alloc::format!("{}/status/last_commit", surface_root), "0\n");
    write_text(&alloc::format!("{}/status/configured_serial", surface_root), "0\n");
    write_text(
        &alloc::format!("{}/status/width", surface_root),
        &alloc::format!("{}\n", WIDTH),
    );
    write_text(
        &alloc::format!("{}/status/height", surface_root),
        &alloc::format!("{}\n", HEIGHT),
    );
    write_text(&alloc::format!("{}/status/buffer_attached", surface_root), "0\n");
}

fn create_buffer() -> BufferState {
    let size = STRIDE as usize * HEIGHT as usize;
    let fd = memfd_create("fetchd.surface", size).expect("memfd");
    let req = VmMapReq {
        addr_hint: 0,
        len: size,
        prot: VmProt::READ | VmProt::WRITE | VmProt::USER,
        flags: VmMapFlags::empty(),
        backing: VmBacking::File { fd, offset: 0 },
    };
    let resp = vm_map(&req).expect("vm_map");
    BufferState {
        fd,
        ptr: resp.addr as *mut u32,
    }
}

fn publish_surface(buffer: &BufferState, commit: u64) {
    let surface_root = alloc::format!("{}/{}", SURFACES_ROOT, SURFACE_ID);
    let attach = alloc::format!(
        "fd={}\nwidth={}\nheight={}\nstride={}\nformat=1\n",
        buffer.fd, WIDTH, HEIGHT, STRIDE
    );
    write_text(&alloc::format!("{}/attach", surface_root), &attach);
    write_text(
        &alloc::format!("{}/damage", surface_root),
        &alloc::format!("0 0 {} {}\n", WIDTH, HEIGHT),
    );
    write_text(
        &alloc::format!("{}/commit", surface_root),
        &alloc::format!("{}\n", commit),
    );
}

fn sync_configure_status() {
    let window_root = alloc::format!("{}/{}", WINDOWS_ROOT, WINDOW_ID);
    let surface_root = alloc::format!("{}/{}", SURFACES_ROOT, SURFACE_ID);
    let serial = read_text(&alloc::format!("{}/status/last_configure_serial", window_root));
    if !serial.is_empty() {
        write_text(
            &alloc::format!("{}/status/configured_serial", surface_root),
            &alloc::format!("{}\n", serial),
        );
    }
}

fn ip_to_string(packed: u64) -> alloc::string::String {
    let a = (packed & 0xFF) as u8;
    let b = ((packed >> 8) & 0xFF) as u8;
    let c = ((packed >> 16) & 0xFF) as u8;
    let d = ((packed >> 24) & 0xFF) as u8;
    alloc::format!("{}.{}.{}.{}", a, b, c, d)
}

fn fill(pixels: &mut [u32], color: u32) {
    for px in pixels.iter_mut() {
        *px = color;
    }
}

fn fill_rect(pixels: &mut [u32], width: usize, x: i32, y: i32, w: i32, h: i32, color: u32) {
    for yy in 0..h {
        let py = y + yy;
        if py < 0 || py >= HEIGHT as i32 {
            continue;
        }
        for xx in 0..w {
            let px = x + xx;
            if px < 0 || px >= WIDTH as i32 {
                continue;
            }
            pixels[py as usize * width + px as usize] = color;
        }
    }
}

fn draw_segment_digit(
    pixels: &mut [u32],
    width: usize,
    x: i32,
    y: i32,
    scale: i32,
    digit: char,
    color: u32,
) {
    const SEGMENTS: [u8; 10] = [
        0b1111110, 0b0110000, 0b1101101, 0b1111001, 0b0110011, 0b1011011, 0b1011111, 0b1110000,
        0b1111111, 0b1111011,
    ];
    let Some(n) = digit.to_digit(10) else {
        return;
    };
    let mask = SEGMENTS[n as usize];
    let t = scale;
    let l = scale * 6;
    let h = scale * 9;

    if mask & 0b1000000 != 0 {
        fill_rect(pixels, width, x + t, y, l, t, color);
    }
    if mask & 0b0100000 != 0 {
        fill_rect(pixels, width, x + l + t, y + t, t, h, color);
    }
    if mask & 0b0010000 != 0 {
        fill_rect(pixels, width, x + l + t, y + h + 2 * t, t, h, color);
    }
    if mask & 0b0001000 != 0 {
        fill_rect(pixels, width, x + t, y + 2 * h + 2 * t, l, t, color);
    }
    if mask & 0b0000100 != 0 {
        fill_rect(pixels, width, x, y + h + 2 * t, t, h, color);
    }
    if mask & 0b0000010 != 0 {
        fill_rect(pixels, width, x, y + t, t, h, color);
    }
    if mask & 0b0000001 != 0 {
        fill_rect(pixels, width, x + t, y + h + t, l, t, color);
    }
}

fn draw_dot(pixels: &mut [u32], width: usize, x: i32, y: i32, scale: i32, color: u32) {
    fill_rect(pixels, width, x, y, scale * 2, scale * 2, color);
}

fn draw_ip(pixels: &mut [u32], width: usize, x: i32, y: i32, text: &str, color: u32) {
    let scale = 3;
    let digit_w = scale * 8;
    let digit_h = scale * 20;
    let mut cursor = x;
    for ch in text.chars() {
        match ch {
            '0'..='9' => {
                draw_segment_digit(pixels, width, cursor, y, scale, ch, color);
                cursor += digit_w + scale;
            }
            '.' => {
                draw_dot(pixels, width, cursor, y + digit_h - scale * 2, scale, color);
                cursor += scale * 4;
            }
            _ => {
                cursor += scale * 4;
            }
        }
    }
}

fn render_window(buffer: &BufferState, ip_text: &str, connected: bool) {
    let pixels = unsafe {
        core::slice::from_raw_parts_mut(buffer.ptr, (WIDTH as usize) * (HEIGHT as usize))
    };
    fill(pixels, 0xFF10151C);
    fill_rect(pixels, WIDTH as usize, 0, 0, WIDTH as i32, 28, 0xFF243445);
    fill_rect(
        pixels,
        WIDTH as usize,
        14,
        42,
        WIDTH as i32 - 28,
        HEIGHT as i32 - 56,
        0xFF16202A,
    );
    fill_rect(
        pixels,
        WIDTH as usize,
        14,
        HEIGHT as i32 - 18,
        WIDTH as i32 - 28,
        4,
        if connected { 0xFF3CCB7F } else { 0xFFE59A3A },
    );
    draw_ip(
        pixels,
        WIDTH as usize,
        26,
        58,
        ip_text,
        if connected { 0xFFBFE9FF } else { 0xFFFFD19A },
    );
}

fn pick_best_ip() -> (alloc::string::String, bool) {
    // In VFS-native model, we read eth0 status from /net/interfaces/eth0/addr
    let text = read_text("/net/interfaces/eth0/addr");
    if text.is_empty() || text.starts_with("0.0.0.0") {
        (alloc::string::String::from("0.0.0.0"), false)
    } else {
        // text is "192.168.1.50/24"
        let ip = text.split('/').next().unwrap_or("0.0.0.0");
        (alloc::string::String::from(ip), true)
    }
}

#[stem::main]
fn main(_arg: usize) -> ! {
    info!("FETCHD: publishing Janix session window");
    ensure_session_objects();

    let buffer = create_buffer();
    let mut commit = 1u64;
    let mut last_ip = u64::MAX;

    loop {
        let (ip_text, connected) = pick_best_ip();

        let packed = if connected {
            let parts = ip_text
                .split('.')
                .filter_map(|part| part.parse::<u8>().ok())
                .collect::<alloc::vec::Vec<_>>();
            if parts.len() == 4 {
                parts[0] as u64
                    | ((parts[1] as u64) << 8)
                    | ((parts[2] as u64) << 16)
                    | ((parts[3] as u64) << 24)
            } else {
                0
            }
        } else {
            0
        };

        if packed != last_ip {
            last_ip = packed;
            render_window(&buffer, &ip_text, connected);
            publish_surface(&buffer, commit);
            commit = commit.wrapping_add(1);
        }
        sync_configure_status();

        stem::sleep(Duration::from_secs(1));
    }
}
