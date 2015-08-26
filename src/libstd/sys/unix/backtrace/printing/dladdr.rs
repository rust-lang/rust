// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io;
use io::prelude::*;
use libc;

pub fn print(w: &mut Write, idx: isize, addr: *mut libc::c_void,
             _symaddr: *mut libc::c_void) -> io::Result<()> {
    use sys_common::backtrace::{output};
    use intrinsics;
    use ffi::CStr;

    #[repr(C)]
    struct Dl_info {
        dli_fname: *const libc::c_char,
        dli_fbase: *mut libc::c_void,
        dli_sname: *const libc::c_char,
        dli_saddr: *mut libc::c_void,
    }
    extern {
        fn dladdr(addr: *const libc::c_void,
                  info: *mut Dl_info) -> libc::c_int;
    }

    let mut info: Dl_info = unsafe { intrinsics::init() };
    if unsafe { dladdr(addr, &mut info) == 0 } {
        output(w, idx,addr, None)
    } else {
        output(w, idx, addr, Some(unsafe {
            CStr::from_ptr(info.dli_sname).to_bytes()
        }))
    }
}
