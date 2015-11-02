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

use backtrace::{output, output_fileline};
use sys::error::Result;
use ffi::CStr;
use dynamic_lib::DynamicLibrary;
use super::{SymFromAddrFn, SymGetLineFromAddr64Fn, SYMBOL_INFO, MAX_SYM_NAME, IMAGEHLP_LINE64};
use io::prelude::*;
use intrinsics;
use libc;

pub fn print(w: &mut Write, i: isize, addr: u64, dbghelp: &DynamicLibrary, process: libc::HANDLE)
        -> Result<()> {
    let SymFromAddr = sym!(dbghelp, "SymFromAddr", SymFromAddrFn);
    let SymGetLineFromAddr64 = sym!(dbghelp, "SymGetLineFromAddr64", SymGetLineFromAddr64Fn);

    let mut info: SYMBOL_INFO = unsafe { intrinsics::init() };
    info.MaxNameLen = MAX_SYM_NAME as libc::c_ulong;
    // the struct size in C.  the value is different to
    // `size_of::<SYMBOL_INFO>() - MAX_SYM_NAME + 1` (== 81)
    // due to struct alignment.
    info.SizeOfStruct = 88;

    let mut displacement = 0u64;
    let ret = SymFromAddr(process, addr, &mut displacement, &mut info);

    let name = if ret == libc::TRUE {
        let ptr = info.Name.as_ptr() as *const libc::c_char;
        Some(unsafe { CStr::from_ptr(ptr).to_bytes() })
    } else {
        None
    };

    try!(output(w, i, addr as usize as *mut (), name));

    // Now find out the filename and line number
    let mut line: IMAGEHLP_LINE64 = unsafe { intrinsics::init() };
    line.SizeOfStruct = ::mem::size_of::<IMAGEHLP_LINE64>() as u32;

    let mut displacement = 0u32;
    let ret = SymGetLineFromAddr64(process, addr, &mut displacement, &mut line);
    if ret == libc::TRUE {
        output_fileline(w,
                        unsafe { CStr::from_ptr(line.Filename).to_bytes() },
                        line.LineNumber as libc::c_int,
                        false)
    } else {
        Ok(())
    }
}
