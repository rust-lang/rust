// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io;
use sys::unsupported;
use sys_common::backtrace::Frame;

pub struct BacktraceContext;

pub fn unwind_backtrace(_frames: &mut [Frame])
    -> io::Result<(usize, BacktraceContext)>
{
    unsupported()
}

pub fn resolve_symname<F>(_frame: Frame,
                          _callback: F,
                          _: &BacktraceContext) -> io::Result<()>
    where F: FnOnce(Option<&str>) -> io::Result<()>
{
    unsupported()
}

pub fn foreach_symbol_fileline<F>(_: Frame,
                                  _: F,
                                  _: &BacktraceContext) -> io::Result<bool>
    where F: FnMut(&[u8], u32) -> io::Result<()>
{
    unsupported()
}
