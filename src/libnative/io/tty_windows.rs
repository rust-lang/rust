// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-lexer-test FIXME #15877

//! Windows specific console TTY implementation
//!
//! This module contains the implementation of a Windows specific console TTY.
//! Also converts between UTF-16 and UTF-8. Windows has very poor support for
//! UTF-8 and some functions will fail. In particular ReadFile and ReadConsole
//! will fail when the codepage is set to UTF-8 and a Unicode character is
//! entered.
//!
//! FIXME
//! This implementation does not account for codepoints that are split across
//! multiple reads and writes. Also, this implementation does not expose a way
//! to read/write UTF-16 directly. When/if Rust receives a Reader/Writer
//! wrapper that performs encoding/decoding, this implementation should switch
//! to working in raw UTF-16, with such a wrapper around it.

use super::c::{ReadConsoleW, WriteConsoleW, GetConsoleMode, SetConsoleMode};
use super::c::{ERROR_ILLEGAL_CHARACTER};
use super::c::{ENABLE_ECHO_INPUT, ENABLE_EXTENDED_FLAGS};
use super::c::{ENABLE_INSERT_MODE, ENABLE_LINE_INPUT};
use super::c::{ENABLE_PROCESSED_INPUT, ENABLE_QUICK_EDIT_MODE};
use libc::{c_int, HANDLE, LPDWORD, DWORD, LPVOID};
use libc::{get_osfhandle, CloseHandle};
use libc::types::os::arch::extra::LPCVOID;
use std::io::MemReader;
use std::ptr;
use std::rt::rtio::{IoResult, IoError, RtioTTY};
use std::str::{from_utf16, from_utf8};

fn invalid_encoding() -> IoError {
    IoError {
        code: ERROR_ILLEGAL_CHARACTER as uint,
        extra: 0,
        detail: Some("text was not valid unicode".to_string()),
    }
}

pub fn is_tty(fd: c_int) -> bool {
    let mut out: DWORD = 0;
    // If this function doesn't fail then fd is a TTY
    match unsafe { GetConsoleMode(get_osfhandle(fd) as HANDLE,
                                  &mut out as LPDWORD) } {
        0 => false,
        _ => true,
    }
}

pub struct WindowsTTY {
    closeme: bool,
    handle: HANDLE,
    utf8: MemReader,
}

impl WindowsTTY {
    pub fn new(fd: c_int) -> WindowsTTY {
        // If the file descriptor is one of stdin, stderr, or stdout
        // then it should not be closed by us
        let closeme = match fd {
            0..2 => false,
            _ => true,
        };
        let handle = unsafe { get_osfhandle(fd) as HANDLE };
        WindowsTTY {
            handle: handle,
            utf8: MemReader::new(Vec::new()),
            closeme: closeme,
        }
    }
}

impl Drop for WindowsTTY {
    fn drop(&mut self) {
        if self.closeme {
            // Nobody cares about the return value
            let _ = unsafe { CloseHandle(self.handle) };
        }
    }
}

impl RtioTTY for WindowsTTY {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        // Read more if the buffer is empty
        if self.utf8.eof() {
            let mut utf16 = Vec::from_elem(0x1000, 0u16);
            let mut num: DWORD = 0;
            match unsafe { ReadConsoleW(self.handle,
                                         utf16.as_mut_ptr() as LPVOID,
                                         utf16.len() as u32,
                                         &mut num as LPDWORD,
                                         ptr::mut_null()) } {
                0 => return Err(super::last_error()),
                _ => (),
            };
            utf16.truncate(num as uint);
            let utf8 = match from_utf16(utf16.as_slice()) {
                Some(utf8) => utf8.into_bytes(),
                None => return Err(invalid_encoding()),
            };
            self.utf8 = MemReader::new(utf8);
        }
        // MemReader shouldn't error here since we just filled it
        Ok(self.utf8.read(buf).unwrap())
    }

    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        let utf16 = match from_utf8(buf) {
            Some(utf8) => utf8.to_utf16(),
            None => return Err(invalid_encoding()),
        };
        let mut num: DWORD = 0;
        match unsafe { WriteConsoleW(self.handle,
                                     utf16.as_ptr() as LPCVOID,
                                     utf16.len() as u32,
                                     &mut num as LPDWORD,
                                     ptr::mut_null()) } {
            0 => Err(super::last_error()),
            _ => Ok(()),
        }
    }

    fn set_raw(&mut self, raw: bool) -> IoResult<()> {
        // FIXME
        // Somebody needs to decide on which of these flags we want
        match unsafe { SetConsoleMode(self.handle,
            match raw {
                true => 0,
                false => ENABLE_ECHO_INPUT | ENABLE_EXTENDED_FLAGS |
                         ENABLE_INSERT_MODE | ENABLE_LINE_INPUT |
                         ENABLE_PROCESSED_INPUT | ENABLE_QUICK_EDIT_MODE,
            }) } {
            0 => Err(super::last_error()),
            _ => Ok(()),
        }
    }

    fn get_winsize(&mut self) -> IoResult<(int, int)> {
        // FIXME
        // Get console buffer via CreateFile with CONOUT$
        // Make a CONSOLE_SCREEN_BUFFER_INFO
        // Call GetConsoleScreenBufferInfo
        // Maybe call GetLargestConsoleWindowSize instead?
        Err(super::unimpl())
    }

    // Let us magically declare this as a TTY
    fn isatty(&self) -> bool { true }
}
