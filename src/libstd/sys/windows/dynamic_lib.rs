// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use os::windows::prelude::*;

use ffi::{CString, OsStr};
use io;
use sys::c;

pub struct DynamicLibrary {
    handle: c::HMODULE,
}

impl DynamicLibrary {
    pub fn open(filename: &str) -> io::Result<DynamicLibrary> {
        let filename = OsStr::new(filename)
                             .encode_wide()
                             .chain(Some(0))
                             .collect::<Vec<_>>();
        let result = unsafe {
            c::LoadLibraryW(filename.as_ptr())
        };
        if result.is_null() {
            Err(io::Error::last_os_error())
        } else {
            Ok(DynamicLibrary { handle: result })
        }
    }

    pub fn symbol(&self, symbol: &str) -> io::Result<usize> {
        let symbol = CString::new(symbol)?;
        unsafe {
            match c::GetProcAddress(self.handle, symbol.as_ptr()) as usize {
                0 => Err(io::Error::last_os_error()),
                n => Ok(n),
            }
        }
    }
}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        unsafe {
            c::FreeLibrary(self.handle);
        }
    }
}
