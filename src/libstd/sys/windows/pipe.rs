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
use ptr;
use sys::cvt;
use sys::c;
use sys::handle::Handle;

////////////////////////////////////////////////////////////////////////////////
// Anonymous pipes
////////////////////////////////////////////////////////////////////////////////

pub struct AnonPipe {
    inner: Handle,
}

pub fn anon_pipe() -> io::Result<(AnonPipe, AnonPipe)> {
    let mut reader = c::INVALID_HANDLE_VALUE;
    let mut writer = c::INVALID_HANDLE_VALUE;
    try!(cvt(unsafe {
        c::CreatePipe(&mut reader, &mut writer, ptr::null_mut(), 0)
    }));
    let reader = Handle::new(reader);
    let writer = Handle::new(writer);
    Ok((AnonPipe { inner: reader }, AnonPipe { inner: writer }))
}

impl AnonPipe {
    pub fn handle(&self) -> &Handle { &self.inner }
    pub fn into_handle(self) -> Handle { self.inner }

    pub fn raw(&self) -> c::HANDLE { self.inner.raw() }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }
}
