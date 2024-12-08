//! Windows console handling

// FIXME (#13400): this is only a tiny fraction of the Windows console api

use std::io;
use std::io::prelude::*;

use super::{Terminal, color};

/// A Terminal implementation that uses the Win32 Console API.
pub(crate) struct WinConsole<T> {
    buf: T,
    def_foreground: color::Color,
    def_background: color::Color,
    foreground: color::Color,
    background: color::Color,
}

type SHORT = i16;
type WORD = u16;
type DWORD = u32;
type BOOL = i32;
type HANDLE = *mut u8;
// https://docs.microsoft.com/en-us/windows/console/getstdhandle
const STD_OUTPUT_HANDLE: DWORD = -11 as _;

#[allow(non_snake_case)]
#[repr(C)]
struct SMALL_RECT {
    Left: SHORT,
    Top: SHORT,
    Right: SHORT,
    Bottom: SHORT,
}

#[allow(non_snake_case)]
#[repr(C)]
struct COORD {
    X: SHORT,
    Y: SHORT,
}

#[allow(non_snake_case)]
#[repr(C)]
struct CONSOLE_SCREEN_BUFFER_INFO {
    dwSize: COORD,
    dwCursorPosition: COORD,
    wAttributes: WORD,
    srWindow: SMALL_RECT,
    dwMaximumWindowSize: COORD,
}

#[allow(non_snake_case)]
#[link(name = "kernel32")]
extern "system" {
    fn SetConsoleTextAttribute(handle: HANDLE, attr: WORD) -> BOOL;
    fn GetStdHandle(which: DWORD) -> HANDLE;
    fn GetConsoleScreenBufferInfo(handle: HANDLE, info: *mut CONSOLE_SCREEN_BUFFER_INFO) -> BOOL;
}

fn color_to_bits(color: color::Color) -> u16 {
    // magic numbers from mingw-w64's wincon.h

    let bits = match color % 8 {
        color::BLACK => 0,
        color::BLUE => 0x1,
        color::GREEN => 0x2,
        color::RED => 0x4,
        color::YELLOW => 0x2 | 0x4,
        color::MAGENTA => 0x1 | 0x4,
        color::CYAN => 0x1 | 0x2,
        color::WHITE => 0x1 | 0x2 | 0x4,
        _ => unreachable!(),
    };

    if color >= 8 { bits | 0x8 } else { bits }
}

fn bits_to_color(bits: u16) -> color::Color {
    let color = match bits & 0x7 {
        0 => color::BLACK,
        0x1 => color::BLUE,
        0x2 => color::GREEN,
        0x4 => color::RED,
        0x6 => color::YELLOW,
        0x5 => color::MAGENTA,
        0x3 => color::CYAN,
        0x7 => color::WHITE,
        _ => unreachable!(),
    };

    color | (u32::from(bits) & 0x8) // copy the hi-intensity bit
}

impl<T: Write + Send + 'static> WinConsole<T> {
    fn apply(&mut self) {
        let _unused = self.buf.flush();
        let mut accum: WORD = 0;
        accum |= color_to_bits(self.foreground);
        accum |= color_to_bits(self.background) << 4;

        unsafe {
            // You may be wondering, "but what about stderr?", and the answer
            // to that is that setting terminal attributes on the stdout
            // handle also sets them for stderr, since they go to the same
            // terminal! Admittedly, this is fragile, since stderr could be
            // redirected to a different console. This is good enough for
            // rustc though. See #13400.
            let out = GetStdHandle(STD_OUTPUT_HANDLE);
            SetConsoleTextAttribute(out, accum);
        }
    }

    pub(crate) fn new(out: T) -> WinConsole<T> {
        use std::mem::MaybeUninit;

        let fg;
        let bg;
        unsafe {
            let mut buffer_info = MaybeUninit::<CONSOLE_SCREEN_BUFFER_INFO>::uninit();
            let handle = GetStdHandle(STD_OUTPUT_HANDLE);
            if GetConsoleScreenBufferInfo(handle, buffer_info.as_mut_ptr()) != 0 {
                let buffer_info = buffer_info.assume_init();
                fg = bits_to_color(buffer_info.wAttributes);
                bg = bits_to_color(buffer_info.wAttributes >> 4);
            } else {
                fg = color::WHITE;
                bg = color::BLACK;
            }
        }
        WinConsole {
            buf: out,
            def_foreground: fg,
            def_background: bg,
            foreground: fg,
            background: bg,
        }
    }
}

impl<T: Write> Write for WinConsole<T> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buf.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.buf.flush()
    }
}

impl<T: Write + Send + 'static> Terminal for WinConsole<T> {
    fn fg(&mut self, color: color::Color) -> io::Result<bool> {
        self.foreground = color;
        self.apply();

        Ok(true)
    }

    fn reset(&mut self) -> io::Result<bool> {
        self.foreground = self.def_foreground;
        self.background = self.def_background;
        self.apply();

        Ok(true)
    }
}
