// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
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

use std::io::prelude::*;
use std::io;
use std::ptr;
use std::sys::windows::c as kernel32;
use std::sys::windows::c as winapi;

use Attr;
use Error;
use Result;
use Terminal;
use color;

/// A Terminal implementation which uses the Win32 Console API.
pub struct WinConsole<T> {
    buf: T,
    def_foreground: color::Color,
    def_background: color::Color,
    foreground: color::Color,
    background: color::Color,
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

// Just get a handle to the current console buffer whatever it is
fn conout() -> io::Result<winapi::HANDLE> {
    let name = b"CONOUT$\0";
    let handle = unsafe {
        kernel32::CreateFileA(name.as_ptr() as *const i8,
                              winapi::GENERIC_READ | winapi::GENERIC_WRITE,
                              winapi::FILE_SHARE_WRITE,
                              ptr::null_mut(),
                              winapi::OPEN_EXISTING,
                              0,
                              ptr::null_mut())
    };
    if handle == winapi::INVALID_HANDLE_VALUE {
        Err(io::Error::last_os_error())
    } else {
        Ok(handle)
    }
}

// This test will only pass if it is running in an actual console, probably
#[test]
fn test_conout() {
    assert!(conout().is_ok())
}

impl<T: Write + Send> WinConsole<T> {
    fn apply(&mut self) -> io::Result<()> {
        let out = try!(conout());
        let _unused = self.buf.flush();
        let mut accum: winapi::WORD = 0;
        accum |= color_to_bits(self.foreground);
        accum |= color_to_bits(self.background) << 4;
        unsafe {
            kernel32::SetConsoleTextAttribute(out, accum);
        }
        Ok(())
    }

    /// Returns `Err` whenever the terminal cannot be created for some
    /// reason.
    pub fn new(out: T) -> io::Result<WinConsole<T>> {
        let fg;
        let bg;
        let handle = try!(conout());
        unsafe {
            let mut buffer_info = ::std::mem::uninitialized();
            if kernel32::GetConsoleScreenBufferInfo(handle, &mut buffer_info) != 0 {
                fg = bits_to_color(buffer_info.wAttributes);
                bg = bits_to_color(buffer_info.wAttributes >> 4);
            } else {
                return Err(io::Error::last_os_error());
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

impl<T: Write + Send> Terminal for WinConsole<T> {
    type Output = T;

    fn fg(&mut self, color: color::Color) -> Result<()> {
        self.foreground = color;
        try!(self.apply());

        Ok(())
    }

    fn bg(&mut self, color: color::Color) -> Result<()> {
        self.background = color;
        try!(self.apply());

        Ok(())
    }

    fn attr(&mut self, attr: Attr) -> Result<()> {
        match attr {
            Attr::ForegroundColor(f) => {
                self.foreground = f;
                try!(self.apply());
                Ok(())
            }
            Attr::BackgroundColor(b) => {
                self.background = b;
                try!(self.apply());
                Ok(())
            }
            _ => Err(Error::NotSupported),
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

    fn reset(&mut self) -> Result<()> {
        self.foreground = self.def_foreground;
        self.background = self.def_background;
        try!(self.apply());

        Ok(())
    }

    fn supports_reset(&self) -> bool {
        true
    }

    fn supports_color(&self) -> bool {
        true
    }

    fn cursor_up(&mut self) -> Result<()> {
        let _unused = self.buf.flush();
        let handle = try!(conout());
        unsafe {
            let mut buffer_info = ::std::mem::uninitialized();
            if kernel32::GetConsoleScreenBufferInfo(handle, &mut buffer_info) != 0 {
                let (x, y) = (buffer_info.dwCursorPosition.X,
                              buffer_info.dwCursorPosition.Y);
                if y == 0 {
                    // Even though this might want to be a CursorPositionInvalid, on Unix there
                    // is no checking to see if the cursor is already on the first line.
                    // I'm not sure what the ideal behavior is, but I think it'd be silly to have
                    // cursor_up fail in this case.
                    Ok(())
                } else {
                    let pos = winapi::COORD {
                        X: x,
                        Y: y - 1,
                    };
                    if kernel32::SetConsoleCursorPosition(handle, pos) != 0 {
                        Ok(())
                    } else {
                        Err(io::Error::last_os_error().into())
                    }
                }
            } else {
                Err(io::Error::last_os_error().into())
            }
        }
    }

    fn delete_line(&mut self) -> Result<()> {
        let _unused = self.buf.flush();
        let handle = try!(conout());
        unsafe {
            let mut buffer_info = ::std::mem::uninitialized();
            if kernel32::GetConsoleScreenBufferInfo(handle, &mut buffer_info) == 0 {
                return Err(io::Error::last_os_error().into());
            }
            let pos = buffer_info.dwCursorPosition;
            let size = buffer_info.dwSize;
            let num = (size.X - pos.X) as winapi::DWORD;
            let mut written = 0;
            if kernel32::FillConsoleOutputCharacterW(handle, 0, num, pos, &mut written) == 0 {
                return Err(io::Error::last_os_error().into());
            }
            if kernel32::FillConsoleOutputAttribute(handle, 0, num, pos, &mut written) == 0 {
                return Err(io::Error::last_os_error().into());
            }
            // Similar reasoning for not failing as in cursor_up -- it doesn't even make
            // sense to
            // me that these APIs could have written 0, unless the terminal is width zero.
            Ok(())
        }
    }

    fn carriage_return(&mut self) -> Result<()> {
        let _unused = self.buf.flush();
        let handle = try!(conout());
        unsafe {
            let mut buffer_info = ::std::mem::uninitialized();
            if kernel32::GetConsoleScreenBufferInfo(handle, &mut buffer_info) != 0 {
                let winapi::COORD { X: x, Y: y } = buffer_info.dwCursorPosition;
                if x == 0 {
                    Err(Error::CursorDestinationInvalid)
                } else {
                    let pos = winapi::COORD {
                        X: 0,
                        Y: y,
                    };
                    if kernel32::SetConsoleCursorPosition(handle, pos) != 0 {
                        Ok(())
                    } else {
                        Err(io::Error::last_os_error().into())
                    }
                }
            } else {
                Err(io::Error::last_os_error().into())
            }
        }
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
