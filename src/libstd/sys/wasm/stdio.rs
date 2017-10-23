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
use sys::{Void, unsupported};

pub struct Stdin(Void);
pub struct Stdout;
pub struct Stderr;

impl Stdin {
    pub fn new() -> io::Result<Stdin> {
        unsupported()
    }

    pub fn read(&self, _data: &mut [u8]) -> io::Result<usize> {
        match self.0 {}
    }
}

impl Stdout {
    pub fn new() -> io::Result<Stdout> {
        Ok(Stdout)
    }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        // If runtime debugging is enabled at compile time we'll invoke some
        // runtime functions that are defined in our src/etc/wasm32-shim.js
        // debugging script. Note that this ffi function call is intended
        // *purely* for debugging only and should not be relied upon.
        if !super::DEBUG {
            return unsupported()
        }
        extern {
            fn rust_wasm_write_stdout(data: *const u8, len: usize);
        }
        unsafe {
            rust_wasm_write_stdout(data.as_ptr(), data.len())
        }
        Ok(data.len())
    }

    pub fn flush(&self) -> io::Result<()> {
        Ok(())
    }
}

impl Stderr {
    pub fn new() -> io::Result<Stderr> {
        Ok(Stderr)
    }

    pub fn write(&self, data: &[u8]) -> io::Result<usize> {
        // See comments in stdout for what's going on here.
        if !super::DEBUG {
            return unsupported()
        }
        extern {
            fn rust_wasm_write_stderr(data: *const u8, len: usize);
        }
        unsafe {
            rust_wasm_write_stderr(data.as_ptr(), data.len())
        }
        Ok(data.len())
    }

    pub fn flush(&self) -> io::Result<()> {
        Ok(())
    }
}

impl io::Write for Stderr {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        (&*self).write(data)
    }
    fn flush(&mut self) -> io::Result<()> {
        (&*self).flush()
    }
}

pub const STDIN_BUF_SIZE: usize = 0;

pub fn is_ebadf(_err: &io::Error) -> bool {
    true
}
