// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(deprecated)]

use prelude::v1::*;

use sys::fs::FileDesc;
use libc::{self, c_int, c_ulong};
use old_io::{self, IoResult, IoError};
use sys::c;
use sys_common;

pub struct TTY {
    pub fd: FileDesc,
}

#[cfg(any(target_os = "macos",
          target_os = "ios",
          target_os = "dragonfly",
          target_os = "freebsd",
          target_os = "bitrig",
          target_os = "openbsd"))]
const TIOCGWINSZ: c_ulong = 0x40087468;

#[cfg(any(target_os = "linux", target_os = "android"))]
const TIOCGWINSZ: c_ulong = 0x00005413;

impl TTY {
    pub fn new(fd: c_int) -> IoResult<TTY> {
        if unsafe { libc::isatty(fd) } != 0 {
            Ok(TTY { fd: FileDesc::new(fd, true) })
        } else {
            Err(IoError {
                kind: old_io::MismatchedFileTypeForOperation,
                desc: "file descriptor is not a TTY",
                detail: None,
            })
        }
    }

    pub fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        self.fd.read(buf)
    }
    pub fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        self.fd.write(buf)
    }
    pub fn set_raw(&mut self, _raw: bool) -> IoResult<()> {
        Err(sys_common::unimpl())
    }

    pub fn get_winsize(&mut self) -> IoResult<(isize, isize)> {
        unsafe {
            #[repr(C)]
            struct winsize {
                ws_row: u16,
                ws_col: u16,
                ws_xpixel: u16,
                ws_ypixel: u16
            }

            let mut size = winsize { ws_row: 0, ws_col: 0, ws_xpixel: 0, ws_ypixel: 0 };
            if c::ioctl(self.fd.fd(), TIOCGWINSZ, &mut size) == -1 {
                Err(IoError {
                    kind: old_io::OtherIoError,
                    desc: "Size of terminal could not be determined",
                    detail: None,
                })
            } else {
                Ok((size.ws_col as isize, size.ws_row as isize))
            }
        }
    }
}
