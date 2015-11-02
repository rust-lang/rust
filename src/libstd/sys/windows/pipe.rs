// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io;
use libc;
use ptr;
use sys::windows::c::{self, cvt};
use sys::windows::handle::Handle;
use sys::error::Result;
use sys::inner::*;

////////////////////////////////////////////////////////////////////////////////
// Anonymous pipes
////////////////////////////////////////////////////////////////////////////////

pub struct AnonPipe(Handle);

impl_inner!(AnonPipe(Handle));

pub fn anon_pipe() -> Result<(AnonPipe, AnonPipe)> {
    let mut reader = libc::INVALID_HANDLE_VALUE;
    let mut writer = libc::INVALID_HANDLE_VALUE;
    try!(cvt(unsafe {
        c::CreatePipe(&mut reader, &mut writer, ptr::null_mut(), 0)
    }));
    let reader = Handle::from_inner(reader);
    let writer = Handle::from_inner(writer);
    Ok((AnonPipe(reader), AnonPipe(writer)))
}

impl io::Read for AnonPipe {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf).map_err(From::from)
    }
}

impl io::Write for AnonPipe {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf).map_err(From::from)
    }

    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}
