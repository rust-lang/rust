// Copyright 2014-2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod dladdr;

use sys::backtrace::BacktraceContext;
use sys_common::backtrace::Frame;
use io;

#[cfg(target_os = "emscripten")]
pub use self::dladdr::resolve_symname;

#[cfg(target_os = "emscripten")]
pub fn foreach_symbol_fileline<F>(_: Frame, _: F, _: &BacktraceContext) -> io::Result<bool>
where
    F: FnMut(&[u8], u32) -> io::Result<()>
{
    Ok(false)
}

#[cfg(not(target_os = "emscripten"))]
pub use sys_common::gnu::libbacktrace::foreach_symbol_fileline;

#[cfg(not(target_os = "emscripten"))]
pub fn resolve_symname<F>(frame: Frame, callback: F, bc: &BacktraceContext) -> io::Result<()>
where
    F: FnOnce(Option<&str>) -> io::Result<()>
{
    ::sys_common::gnu::libbacktrace::resolve_symname(frame, |symname| {
        if symname.is_some() {
            callback(symname)
        } else {
            dladdr::resolve_symname(frame, callback, bc)
        }
    }, bc)
}
