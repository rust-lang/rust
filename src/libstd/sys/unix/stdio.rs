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
use sys::fd::FileDesc;
use sys::fs::{File, OpenOptions};
use ffi::CStr;

pub struct Stdin(());
pub struct Stdout(());
pub struct Stderr(());

// FIXME: This duplicates code from process_common.rs.
fn open_null_device (readable: bool) -> io::Result<FileDesc> {
    let mut opts = OpenOptions::new();
    opts.read(readable);
    opts.write(!readable);
    let path = unsafe {
        CStr::from_ptr("/dev/null\0".as_ptr() as *const _)
    };
    let fd = File::open_c(&path, &opts)?;
    Ok(fd.into_fd())
}

impl Stdin {
    pub fn new() -> io::Result<Stdin> { Ok(Stdin(())) }

    pub fn read(&self, data: &mut [u8]) -> io::Result<usize> {
        let fd = FileDesc::new(libc::STDIN_FILENO);
        let ret = fd.read(data);
        fd.into_raw();
        ret
    }

    pub fn close() -> io::Result<()> {
        // To close stdin, what we actually do is change its well-known
        // file descriptor number to refer to a file open on the null
        // device.  This protects against code (perhaps in third-party
        // libraries) that assumes STDIN_FILENO is always open and always
        // refers to the same thing as stdin.
        let mut fd = FileDesc::new(libc::STDIN_FILENO);
        // If this step succeeds, the "previous" file descriptor returned
        // by `fd.replace` is dropped and thus closed.
        fd.replace(open_null_device(true)?)?;
        // Don't close STDIN_FILENO itself, though!
        fd.into_raw();
        Ok(())
    }
}

impl Stdout {
    pub fn new() -> io::Result<Stdout> { Ok(Stdout(())) }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        let fd = FileDesc::new(libc::STDOUT_FILENO);
        let ret = fd.write(data);
        fd.into_raw();
        ret
    }

    pub fn flush(&self) -> io::Result<()> {
        Ok(())
    }

    pub fn close() -> io::Result<()> {
        // See commentary for Stdin::close.

        let mut fd = FileDesc::new(libc::STDOUT_FILENO);
        fd.replace(open_null_device(false)?)?;
        fd.into_raw();
        Ok(())
    }
}

impl Stderr {
    pub fn new() -> io::Result<Stderr> { Ok(Stderr(())) }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        let fd = FileDesc::new(libc::STDERR_FILENO);
        let ret = fd.write(data);
        fd.into_raw();
        ret
    }

    pub fn flush(&self) -> io::Result<()> {
        Ok(())
    }

    pub fn close() -> io::Result<()> {
        // See commentary for Stdin::close.

        let mut fd = FileDesc::new(libc::STDERR_FILENO);
        fd.replace(open_null_device(false)?)?;
        fd.into_raw();
        Ok(())
    }
}

// FIXME: right now this raw stderr handle is used in a few places because
//        std::io::stderr_raw isn't exposed, but once that's exposed this impl
//        should go away
impl io::Write for Stderr {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        Stderr::write(self, data)
    }

    fn flush(&mut self) -> io::Result<()> {
        Stderr::flush(self)
    }
}

pub fn is_ebadf(err: &io::Error) -> bool {
    err.raw_os_error() == Some(libc::EBADF as i32)
}

pub const STDIN_BUF_SIZE: usize = ::sys_common::io::DEFAULT_BUF_SIZE;
