#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;


use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

use abi::display_driver_protocol::{BindPayload, FbInfoPayload, FB_INFO_PAYLOAD_SIZE};
use abi::syscall::vfs_flags::{O_CREAT, O_RDONLY, O_WRONLY};
use abi::vfs_watch::mask;
use stem::syscall::vfs::{vfs_close, vfs_open, vfs_read, vfs_stat, vfs_watch_path, vfs_write};
use stem::{error, info};

/// A single glyph from the Unifont font.
struct Glyph {
    width: u32,
    bitmap: Vec<u8>,
}

struct Font {
    glyphs: BTreeMap<u32, Glyph>,
}

impl Font {
    fn load(path: &str) -> Result<Self, String> {
        let fd =
            vfs_open(path, O_RDONLY).map_err(|e| alloc::format!("failed to open font file: {:?}", e))?;
        let stat = vfs_stat(fd).map_err(|e| alloc::format!("failed to stat font file: {:?}", e))?;
        let size = stat.size;

        let mut data = Vec::with_capacity(size as usize);
        data.resize(size as usize, 0);
        let n =
            vfs_read(fd, &mut data).map_err(|e| alloc::format!("failed to read font file: {:?}", e))?;
        data.truncate(n);
        let _ = vfs_close(fd);

        let content = String::from_utf8_lossy(&data);
        let mut glyphs = BTreeMap::new();

        for line in content.lines() {
            if let Some((code_str, bitmap_str)) = line.split_once(':') {
                if let Ok(code) = u32::from_str_radix(code_str, 16) {
                    let mut bitmap = Vec::new();
                    for i in 0..(bitmap_str.len() / 2) {
                        if let Ok(byte) = u8::from_str_radix(&bitmap_str[i * 2..i * 2 + 2], 16) {
                            bitmap.push(byte);
                        }
                    }

                    let width = if bitmap_str.len() <= 32 { 8 } else { 16 };
                    glyphs.insert(code, Glyph { width, bitmap });
                }
            }
        }

        info!("Terminal: Loaded {} glyphs from {}", glyphs.len(), path);
        Ok(Font { glyphs })
    }

    fn get_glyph(&self, c: char) -> Option<&Glyph> {
        self.glyphs.get(&(c as u32))
    }
}

#[derive(Debug, Clone, PartialEq)]
enum AnsiState {
    Normal,
    Esc,
    Csi { params: Vec<u32>, current_num: Option<u32> },
}

struct Terminal {
    width: u32,
    height: u32,
    stride: u32,
    fb_ptr: *mut u32,
    font: Font,
    cursor_x: u32,
    cursor_y: u32,
    current_fg: u32,
    current_bg: u32,
    ansi_state: AnsiState,
}

impl Terminal {
    fn new(info: FbInfoPayload, font: Font, fb_ptr: *mut u32) -> Self {
        Self {
            width: info.width,
            height: info.height,
            stride: info.stride,
            fb_ptr,
            font,
            cursor_x: 0,
            cursor_y: 0,
            current_fg: 0xFFFFFFFF, // White
            current_bg: 0xFF000000, // Black
            ansi_state: AnsiState::Normal,
        }
    }

    fn clear(&mut self, color: u32) {
        let size = (self.stride / 4) * self.height;
        unsafe {
            let slice = core::slice::from_raw_parts_mut(self.fb_ptr, size as usize);
            slice.fill(color);
        }
    }

    fn ansi_color_to_u32(code: u32, is_bg: bool) -> u32 {
        let base = if is_bg { 40 } else { 30 };
        match code - base {
            0 => 0xFF000000, // Black
            1 => 0xFFFF0000, // Red
            2 => 0xFF00FF00, // Green
            3 => 0xFFFFFF00, // Yellow
            4 => 0xFF0000FF, // Blue
            5 => 0xFFFF00FF, // Magenta
            6 => 0xFF00FFFF, // Cyan
            7 => 0xFFFFFFFF, // White
            _ => if is_bg { 0xFF000000 } else { 0xFFFFFFFF },
        }
    }

    fn putc(&mut self, c: char) {
        match self.ansi_state.clone() {
            AnsiState::Normal => {
                if c == '\x1B' {
                    self.ansi_state = AnsiState::Esc;
                    return;
                }
                if c == '\n' {
                    self.cursor_x = 0;
                    self.cursor_y += 16;
                    if self.cursor_y + 16 > self.height {
                        self.scroll();
                    }
                    return;
                }
                if c == '\r' {
                    self.cursor_x = 0;
                    return;
                }
                if c == '\x08' { // Backspace
                     let width = 8; // Assuming standard width for backspace for now
                     if self.cursor_x >= width {
                         self.cursor_x -= width;
                     }
                     return;
                }
                // Handle tab as 4 spaces
                if c == '\t' {
                    for _ in 0..4 {
                        self.putc(' ');
                    }
                    return;
                }
            }
            AnsiState::Esc => {
                if c == '[' {
                    self.ansi_state = AnsiState::Csi { params: Vec::new(), current_num: None };
                } else {
                    self.ansi_state = AnsiState::Normal;
                }
                return;
            }
            AnsiState::Csi { mut params, mut current_num } => {
                if c.is_ascii_digit() {
                    let digit = c.to_digit(10).unwrap();
                    current_num = Some(current_num.unwrap_or(0) * 10 + digit);
                    self.ansi_state = AnsiState::Csi { params, current_num };
                    return;
                } else if c == ';' {
                    params.push(current_num.unwrap_or(0));
                    self.ansi_state = AnsiState::Csi { params, current_num: None };
                    return;
                } else if c == 'm' {
                    // SGR - Select Graphic Rendition
                    params.push(current_num.unwrap_or(0));
                    for &p in &params {
                        if p == 0 {
                            self.current_fg = 0xFFFFFFFF;
                            self.current_bg = 0xFF000000;
                        } else if (30..=37).contains(&p) {
                            self.current_fg = Self::ansi_color_to_u32(p, false);
                        } else if (40..=47).contains(&p) {
                            self.current_bg = Self::ansi_color_to_u32(p, true);
                        } else if (90..=97).contains(&p) {
                            // Bright fg
                             self.current_fg = Self::ansi_color_to_u32(p - 60, false) | 0xFF888888; // Hacky bright
                        }
                    }
                    self.ansi_state = AnsiState::Normal;
                    return;
                } else if c == 'J' {
                    // Clear screen
                    self.clear(self.current_bg);
                    self.ansi_state = AnsiState::Normal;
                    return;
                } else if c == 'H' {
                    // Cursor home
                    self.cursor_x = 0;
                    self.cursor_y = 0;
                    self.ansi_state = AnsiState::Normal;
                    return;
                } else {
                    // Unsupported CSI, just go back to normal
                    self.ansi_state = AnsiState::Normal;
                    return;
                }
            }
        }

        let width = self
            .font
            .get_glyph(c)
            .map(|g| g.width)
            .or_else(|| self.font.get_glyph('?').map(|g| g.width))
            .unwrap_or(8);

        if self.cursor_x + width > self.width {
            self.putc('\n');
        }

        let bitmap = self
            .font
            .get_glyph(c)
            .or_else(|| self.font.get_glyph('?'))
            .map(|g| g.bitmap.clone());

        if let Some(bitmap) = bitmap {
            self.draw_glyph_internal(
                bitmap.as_slice(),
                width,
                self.cursor_x,
                self.cursor_y,
                self.current_fg,
                self.current_bg,
            );
            self.cursor_x += width;
        }
    }

    fn draw_glyph_internal(&mut self, bitmap: &[u8], width: u32, x: u32, y: u32, fg: u32, bg: u32) {
        if bitmap.len() == 16 {
            // 8x16
            for row in 0..16 {
                let bits = bitmap[row];
                for col in 0..8 {
                    let color = if (bits & (0x80 >> col)) != 0 { fg } else { bg };
                    self.set_pixel(x + col as u32, y + row as u32, color);
                }
            }
        } else if bitmap.len() == 32 {
            // 16x16
            for row in 0..16 {
                let b1 = bitmap[row * 2];
                let b2 = bitmap[row * 2 + 1];
                for col in 0..8 {
                    let color = if (b1 & (0x80 >> col)) != 0 { fg } else { bg };
                    self.set_pixel(x + col as u32, y + row as u32, color);
                    let color2 = if (b2 & (0x80 >> col)) != 0 { fg } else { bg };
                    self.set_pixel(x + col as u32 + 8, y + row as u32, color2);
                }
            }
        }
    }

    fn set_pixel(&mut self, x: u32, y: u32, color: u32) {
        if x < self.width && y < self.height {
            unsafe {
                let off = y * (self.stride / 4) + x;
                *self.fb_ptr.add(off as usize) = color;
            }
        }
    }

    fn scroll(&mut self) {
        let stride_pixels = (self.stride / 4) as usize;
        let row_pixels = 16 * stride_pixels;
        let total_pixels = (self.height as usize) * stride_pixels;

        unsafe {
            core::ptr::copy(
                self.fb_ptr.add(row_pixels),
                self.fb_ptr,
                total_pixels - row_pixels,
            );
            let last_lines = core::slice::from_raw_parts_mut(
                self.fb_ptr.add(total_pixels - row_pixels),
                row_pixels,
            );
            last_lines.fill(self.current_bg);
        }
        self.cursor_y -= 16;
    }

    fn write_str(&mut self, s: &str) {
        for c in s.chars() {
            self.putc(c);
        }
    }
}

fn get_active_ui() -> String {
    if let Ok(fd) = vfs_open("/session/active_ui", O_RDONLY) {
        if let Ok(stat) = vfs_stat(fd) {
            let size = stat.size;
            let mut buf = Vec::with_capacity(size as usize);
            buf.resize(size as usize, 0);
            if let Ok(n) = vfs_read(fd, &mut buf) {
                buf.truncate(n);
                let _ = vfs_close(fd);
                return String::from_utf8_lossy(&buf).trim().to_string();
            }
        }
        let _ = vfs_close(fd);
    }
    "terminal".to_string()
}

#[stem::main]
fn main(arg: usize) -> ! {
    info!("Terminal: Starting...");

    let boot_fd = arg as u32;
    let mut display_req_write = 0u32;
    let mut display_resp_read = 0u32;
    let mut fb_id = 0u32;

    if boot_fd != 0 {
        use abi::vm::{VmBacking, VmMapFlags, VmMapReq, VmProt};
        let req = VmMapReq {
            addr_hint: 0,
            len: 4096,
            prot: VmProt::READ | VmProt::USER,
            flags: VmMapFlags::empty(),
            backing: VmBacking::File {
                fd: boot_fd,
                offset: 0,
            },
        };
        if let Ok(resp) = stem::syscall::vm_map(&req) {
            let ptr = resp.addr as *const u32;
            let slice = unsafe { core::slice::from_raw_parts(ptr, 1024) };
            info!(
                "Terminal: slice[0]=0x{:08x} [1]=0x{:x} [2]=0x{:x} [4]=0x{:x}",
                slice[0], slice[1], slice[2], slice[4]
            );
            if slice[0] == 0xB100AA01 {
                display_req_write = slice[1];
                display_resp_read = slice[2];
                fb_id = slice[4];
                info!(
                    "Terminal: Bootstrapped via memfd: req={}, resp={}, fb_id={}",
                    display_req_write, display_resp_read, fb_id
                );
            } else {
                error!(
                    "Terminal: Bootstrap magic mismatch! expected 0xB100AA01, got 0x{:08x}",
                    slice[0]
                );
            }
        } else {
            error!("Terminal: Failed to map bootstrap memfd");
        }
    } else {
        info!("Terminal: No bootstrap FD provided (arg was 0)");
    }

    if fb_id == 0 {
        info!("Terminal: No bootstrap FB, trying /dev/fb0 fallback...");
        match vfs_open("/dev/fb0", O_RDONLY) {
            Ok(fd) => {
                fb_id = fd;
                info!("Terminal: Using /dev/fb0 as fb_id={}", fb_id);
            }
            Err(e) => {
                error!("Terminal: Failed to open /dev/fb0: {:?}", e);
                stem::syscall::exit(1);
            }
        }
    }

    let font = match Font::load("/share/fonts/unifont.hex") {
        Ok(f) => f,
        Err(e) => {
            error!("Terminal: Failed to load font: {}", e);
            stem::syscall::exit(1);
        }
    };

    let fb_info = match read_fb_info() {
        Some(info) => info,
        None => {
            error!("Terminal: Failed to read fb info from /dev/fb0");
            stem::syscall::exit(1);
        }
    };

    info!(
        "Terminal: Display {}x{}, stride={}",
        fb_info.width, fb_info.height, fb_info.stride
    );

    let fb_ptr = {
        use abi::vm::{VmBacking, VmMapFlags, VmMapReq, VmProt};
        let req = VmMapReq {
            addr_hint: 0,
            len: (fb_info.stride as usize) * (fb_info.height as usize),
            prot: VmProt::READ | VmProt::WRITE | VmProt::USER,
            flags: VmMapFlags::empty(),
            backing: VmBacking::File {
                fd: fb_id,
                offset: 0,
            },
        };
        match stem::syscall::vm_map(&req) {
            Ok(resp) => resp.addr as *mut u32,
            Err(e) => {
                error!("Terminal: Failed to map FB bytespace {}: {:?}", fb_id, e);
                stem::syscall::exit(1);
            }
        }
    };

    let mut term = Terminal::new(fb_info, font, fb_ptr);
    term.clear(0xFF000000);
    term.write_str("Thing-OS Terminal v1.0\n");
    term.write_str("Unicode test: こんにち世界! 🚀\n");

    // Connect to display driver
    let bind_payload = BindPayload {
        fb_fd: fb_id,
        _pad: 0,
        width: fb_info.width,
        height: fb_info.height,
        stride: fb_info.stride,
        format: fb_info.format,
    };
    if display_req_write != 0 {
        let mut header_buf = [0u8; abi::display_driver_protocol::HEADER_SIZE
            + abi::display_driver_protocol::BIND_PAYLOAD_WIRE_SIZE];
        let mut payload_buf = [0u8; abi::display_driver_protocol::BIND_PAYLOAD_WIRE_SIZE];
        abi::display_driver_protocol::encode_bind_payload_le(&bind_payload, &mut payload_buf);
        if let Some(total) = abi::display_driver_protocol::encode_message(
            &mut header_buf,
            abi::display_driver_protocol::MSG_BIND,
            &payload_buf,
        ) {
            let _ = stem::syscall::channel_send_all(display_req_write, &header_buf[..total]);
            // Transfer the framebuffer FD via the FD-first message queue API.
            let _ = stem::syscall::channel::channel_send_msg(
                display_req_write,
                &[],
                &[bind_payload.fb_fd],
            );
        }
    }

    // Focus handling
    let focus_watch =
        vfs_watch_path("/session/active_ui", abi::vfs_watch::mask::MODIFY, 0).unwrap_or(0);
    let mut has_focus = if display_req_write == 0 {
        true // In fallback mode, we are always active
    } else {
        get_active_ui() == "terminal"
    };

    let mut frame_count = 0u64;
    loop {
        if has_focus && display_req_write != 0 {
            // Present!
            let mut present_header = [0u8; abi::display_driver_protocol::HEADER_SIZE
                + abi::display_driver_protocol::PRESENT_HEADER_WIRE_SIZE];
            let mut payload = [0u8; abi::display_driver_protocol::PRESENT_HEADER_WIRE_SIZE];
            abi::display_driver_protocol::encode_present_header_le(0, &mut payload);
            if let Some(total) = abi::display_driver_protocol::encode_message(
                &mut present_header,
                abi::display_driver_protocol::MSG_PRESENT,
                &payload,
            ) {
                let _ =
                    stem::syscall::channel_send_all(display_req_write, &present_header[..total]);
            }
        }

        if frame_count % 60 == 0 {
            info!("Terminal: Liveness check - frame {}", frame_count);
            term.write_str(".");
            if frame_count % (60 * 40) == 0 {
                let mut status = String::new();
                let _ = write!(
                    status,
                    "\n[Terminal Liveness] Frame {} - Focus: {}\n",
                    frame_count, has_focus
                );
                term.write_str(&status);
            }
        }

        // Check for focus change
        if focus_watch != 0 {
            let mut fds = [abi::syscall::PollFd {
                fd: focus_watch as i32,
                events: abi::syscall::poll_flags::POLLIN as u16,
                revents: 0,
            }];
            if let Ok(n) = stem::syscall::vfs::vfs_poll(&mut fds, 0) {
                if n > 0 {
                    // Read the watch event to clear it
                    let mut dummy = [0u8; 1024];
                    let _ = vfs_read(focus_watch, &mut dummy);

                    let new_focus = get_active_ui() == "terminal";
                    if new_focus != has_focus {
                        has_focus = new_focus;
                        if has_focus {
                            info!("Terminal: Gained focus!");
                        } else {
                            info!("Terminal: Lost focus. Blanking screen.");
                            term.clear(0xFF000000); // Black
                                                    // Send one last present to show the black screen
                            if display_req_write != 0 {
                                let mut present_header = [0u8;
                                    abi::display_driver_protocol::HEADER_SIZE
                                        + abi::display_driver_protocol::PRESENT_HEADER_WIRE_SIZE];
                                let mut payload =
                                    [0u8; abi::display_driver_protocol::PRESENT_HEADER_WIRE_SIZE];
                                abi::display_driver_protocol::encode_present_header_le(
                                    0,
                                    &mut payload,
                                );
                                if let Some(total) = abi::display_driver_protocol::encode_message(
                                    &mut present_header,
                                    abi::display_driver_protocol::MSG_PRESENT,
                                    &payload,
                                ) {
                                    let _ = stem::syscall::channel_send_all(
                                        display_req_write,
                                        &present_header[..total],
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        frame_count += 1;
        stem::sleep(core::time::Duration::from_millis(16)); // ~60fps
    }
}

fn read_fb_info() -> Option<FbInfoPayload> {
    let fd = vfs_open("/dev/fb0", O_RDONLY).ok()?;
    let mut payload = FbInfoPayload {
        device_handle: 0,
        width: 0,
        height: 0,
        stride: 0,
        bpp: 0,
        format: 0,
        _reserved: 0,
    };
    let buf = unsafe {
        core::slice::from_raw_parts_mut(&mut payload as *mut _ as *mut u8, FB_INFO_PAYLOAD_SIZE)
    };
    let n = stem::syscall::vfs::vfs_read(fd, buf).ok()?;
    let _ = stem::syscall::vfs::vfs_close(fd);
    if n < FB_INFO_PAYLOAD_SIZE || payload.width == 0 || payload.height == 0 || payload.stride == 0
    {
        stem::error!(
            "Terminal: FB info mismatch: n={}, expected={}, w={}, h={}, s={}",
            n,
            FB_INFO_PAYLOAD_SIZE,
            payload.width,
            payload.height,
            payload.stride
        );
        return None;
    }
    Some(payload)
}
