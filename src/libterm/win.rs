// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Windows console handling

// FIXME (#13400): this is only a tiny fraction of the Windows console api

extern crate libc;

use std::io;
use std::io::prelude::*;

use Attr;
use color;
use Terminal;

/// A Terminal implementation which uses the Win32 Console API.
pub struct WinConsole<T> {
    buf: T,
    def_foreground: color::Color,
    def_background: color::Color,
    foreground: color::Color,
    background: color::Color,
}

type WORD = u16;
type DWORD = u32;
type BOOL = i32;
type HANDLE = *mut u8;

#[allow(non_snake_case)]
#[repr(C)]
struct CONSOLE_SCREEN_BUFFER_INFO {
    dwSize: [libc::c_short; 2],
    dwCursorPosition: [libc::c_short; 2],
    wAttributes: WORD,
    srWindow: [libc::c_short; 4],
    dwMaximumWindowSize: [libc::c_short; 2],
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

    if color >= 8 {
        bits | 0x8
    } else {
        bits
    }
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

    color | (bits & 0x8) // copy the hi-intensity bit
}

impl<T: Write+Send+'static> WinConsole<T> {
    fn apply(&mut self) {
        let _unused = self.buf.flush();
        let mut accum: WORD = 0;
        accum |= color_to_bits(self.foreground);
        accum |= color_to_bits(self.background) << 4;

        unsafe {
            // Magic -11 means stdout, from
            // http://msdn.microsoft.com/en-us/library/windows/desktop/ms683231%28v=vs.85%29.aspx
            //
            // You may be wondering, "but what about stderr?", and the answer
            // to that is that setting terminal attributes on the stdout
            // handle also sets them for stderr, since they go to the same
            // terminal! Admittedly, this is fragile, since stderr could be
            // redirected to a different console. This is good enough for
            // rustc though. See #13400.
            let out = GetStdHandle(-11i32 as DWORD);
            SetConsoleTextAttribute(out, accum);
        }
    }

    /// Returns `None` whenever the terminal cannot be created for some
    /// reason.
    pub fn new(out: T) -> io::Result<WinConsole<T>> {
        let fg;
        let bg;
        unsafe {
            let mut buffer_info = ::std::mem::uninitialized();
            if GetConsoleScreenBufferInfo(GetStdHandle(-11i32 as DWORD), &mut buffer_info) != 0 {
                fg = bits_to_color(buffer_info.wAttributes);
                bg = bits_to_color(buffer_info.wAttributes >> 4);
            } else {
                fg = color::WHITE;
                bg = color::BLACK;
            }
        }
        Ok(WinConsole {
            buf: out,
            def_foreground: fg,
            def_background: bg,
            foreground: fg,
            background: bg,
        })
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

impl<T: Write+Send+'static> Terminal for WinConsole<T> {
    type Output = T;

    fn fg(&mut self, color: color::Color) -> io::Result<bool> {
        self.foreground = color;
        self.apply();

        Ok(true)
    }

    fn bg(&mut self, color: color::Color) -> io::Result<bool> {
        self.background = color;
        self.apply();

        Ok(true)
    }

    fn attr(&mut self, attr: Attr) -> io::Result<bool> {
        match attr {
            Attr::ForegroundColor(f) => {
                self.foreground = f;
                self.apply();
                Ok(true)
            }
            Attr::BackgroundColor(b) => {
                self.background = b;
                self.apply();
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn supports_attr(&self, attr: Attr) -> bool {
        // it claims support for underscore and reverse video, but I can't get
        // it to do anything -cmr
        match attr {
            Attr::ForegroundColor(_) | Attr::BackgroundColor(_) => true,
            _ => false,
        }
    }

    fn reset(&mut self) -> io::Result<bool> {
        self.foreground = self.def_foreground;
        self.background = self.def_background;
        self.apply();

        Ok(true)
    }

    fn get_ref<'a>(&'a self) -> &'a T {
        &self.buf
    }

    fn get_mut<'a>(&'a mut self) -> &'a mut T {
        &mut self.buf
    }

    fn into_inner(self) -> T
        where Self: Sized
    {
        self.buf
    }
}
