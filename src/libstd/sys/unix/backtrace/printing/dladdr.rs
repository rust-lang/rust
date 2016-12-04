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
use intrinsics;
use ffi::CStr;
use libc;
use sys::backtrace::BacktraceContext;
use sys_common::backtrace::Frame;

pub fn resolve_symname<F>(frame: Frame,
                          callback: F,
                          _: &BacktraceContext) -> io::Result<()>
    where F: FnOnce(Option<&str>) -> io::Result<()>
{
    unsafe {
        let mut info: Dl_info = intrinsics::init();
        let symname = if dladdr(frame.exact_position, &mut info) == 0 {
            None
        } else {
            CStr::from_ptr(info.dli_sname).to_str().ok()
        };
        callback(symname)
    }
}

pub fn foreach_symbol_fileline<F>(_symbol_addr: Frame,
                                  _f: F,
                                  _: &BacktraceContext) -> io::Result<bool>
    where F: FnMut(&[u8], libc::c_int) -> io::Result<()>
{
    Ok(false)
}

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
