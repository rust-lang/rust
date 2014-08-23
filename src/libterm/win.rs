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

use std::io::IoResult;

use attr;
use color;
use Terminal;

/// A Terminal implementation which uses the Win32 Console API.
pub struct WinConsole<T> {
    buf: T,
    foreground: color::Color,
    background: color::Color,
}

#[allow(non_snake_case_functions)]
#[link(name = "kernel32")]
extern "system" {
    fn SetConsoleTextAttribute(handle: libc::HANDLE, attr: libc::WORD) -> libc::BOOL;
    fn GetStdHandle(which: libc::DWORD) -> libc::HANDLE;
}

fn color_to_bits(color: color::Color) -> u16 {
    // magic numbers from mingw-w64's wincon.h

    let bits = match color % 8 {
        color::BLACK   => 0,
        color::BLUE    => 0x1,
        color::GREEN   => 0x2,
        color::RED     => 0x4,
        color::YELLOW  => 0x2 | 0x4,
        color::MAGENTA => 0x1 | 0x4,
        color::CYAN    => 0x1 | 0x2,
        color::WHITE   => 0x1 | 0x2 | 0x4,
        _ => unreachable!()
    };

    if color >= 8 {
        bits | 0x8
    } else {
        bits
    }
}

impl<T: Writer> WinConsole<T> {
    fn apply(&mut self) {
        let _unused = self.buf.flush();
        let mut accum: libc::WORD = 0;
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
            let out = GetStdHandle(-11);
            SetConsoleTextAttribute(out, accum);
        }
    }
}

impl<T: Writer> Writer for WinConsole<T> {
    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        self.buf.write(buf)
    }

    fn flush(&mut self) -> IoResult<()> {
        self.buf.flush()
    }
}

impl<T: Writer> Terminal<T> for WinConsole<T> {
    fn new(out: T) -> Option<WinConsole<T>> {
        Some(WinConsole { buf: out, foreground: color::WHITE, background: color::BLACK })
    }

    fn fg(&mut self, color: color::Color) -> IoResult<bool> {
        self.foreground = color;
        self.apply();

        Ok(true)
    }

    fn bg(&mut self, color: color::Color) -> IoResult<bool> {
        self.background = color;
        self.apply();

        Ok(true)
    }

    fn attr(&mut self, attr: attr::Attr) -> IoResult<bool> {
        match attr {
            attr::ForegroundColor(f) => {
                self.foreground = f;
                self.apply();
                Ok(true)
            },
            attr::BackgroundColor(b) => {
                self.background = b;
                self.apply();
                Ok(true)
            },
            _ => Ok(false)
        }
    }

    fn supports_attr(&self, attr: attr::Attr) -> bool {
        // it claims support for underscore and reverse video, but I can't get
        // it to do anything -cmr
        match attr {
            attr::ForegroundColor(_) | attr::BackgroundColor(_) => true,
            _ => false
        }
    }

    fn reset(&mut self) -> IoResult<()> {
        self.foreground = color::WHITE;
        self.background = color::BLACK;
        self.apply();

        Ok(())
    }

    fn unwrap(self) -> T { self.buf }

    fn get_ref<'a>(&'a self) -> &'a T { &self.buf }

    fn get_mut<'a>(&'a mut self) -> &'a mut T { &mut self.buf }
}
