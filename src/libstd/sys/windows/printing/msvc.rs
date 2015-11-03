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

use dynamic_lib::DynamicLibrary;
use ffi::CStr;
use io::prelude::*;
use io;
use libc::{c_ulong, c_int, c_char, c_void};
use mem;
use super::{SymFromAddrFn, SymGetLineFromAddr64Fn};
use sys::c;
use sys_common::backtrace::{output, output_fileline};

pub fn print(w: &mut Write, i: isize, addr: u64, dbghelp: &DynamicLibrary,
             process: c::HANDLE) -> io::Result<()> {
    let SymFromAddr = sym!(dbghelp, "SymFromAddr", SymFromAddrFn);
    let SymGetLineFromAddr64 = sym!(dbghelp, "SymGetLineFromAddr64", SymGetLineFromAddr64Fn);

    let mut info: c::SYMBOL_INFO = unsafe { mem::zeroed() };
    info.MaxNameLen = c::MAX_SYM_NAME as c_ulong;
    // the struct size in C.  the value is different to
    // `size_of::<SYMBOL_INFO>() - MAX_SYM_NAME + 1` (== 81)
    // due to struct alignment.
    info.SizeOfStruct = 88;

    let mut displacement = 0u64;
    let ret = SymFromAddr(process, addr, &mut displacement, &mut info);

    let name = if ret == c::TRUE {
        let ptr = info.Name.as_ptr() as *const c_char;
        Some(unsafe { CStr::from_ptr(ptr).to_bytes() })
    } else {
        None
    };

    try!(output(w, i, addr as usize as *mut c_void, name));

    // Now find out the filename and line number
    let mut line: c::IMAGEHLP_LINE64 = unsafe { mem::zeroed() };
    line.SizeOfStruct = ::mem::size_of::<c::IMAGEHLP_LINE64>() as u32;

    let mut displacement = 0u32;
    let ret = SymGetLineFromAddr64(process, addr, &mut displacement, &mut line);
    if ret == c::TRUE {
        output_fileline(w,
                        unsafe { CStr::from_ptr(line.Filename).to_bytes() },
                        line.LineNumber as c_int,
                        false)
    } else {
        Ok(())
    }
}
