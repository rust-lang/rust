// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Synchronous, in-memory pipes.
//!
//! Currently these aren't particularly useful, there only exists bindings
//! enough so that pipes can be created to child processes.

use prelude::*;
use super::{Reader, Writer};
use rt::io::{io_error, read_error, EndOfFile};
use rt::local::Local;
use rt::rtio::{RtioPipe, RtioPipeObject, IoFactoryObject, IoFactory};
use rt::rtio::RtioUnboundPipeObject;

pub struct PipeStream {
    priv obj: RtioPipeObject
}

// This should not be a newtype, but rt::uv::process::set_stdio needs to reach
// into the internals of this :(
pub struct UnboundPipeStream(~RtioUnboundPipeObject);

impl PipeStream {
    /// Creates a new pipe initialized, but not bound to any particular
    /// source/destination
    pub fn new() -> Option<UnboundPipeStream> {
        let pipe = unsafe {
            let io: *mut IoFactoryObject = Local::unsafe_borrow();
            (*io).pipe_init(false)
        };
        match pipe {
            Ok(p) => Some(UnboundPipeStream(p)),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
                None
            }
        }
    }

    pub fn bind(inner: RtioPipeObject) -> PipeStream {
        PipeStream { obj: inner }
    }
}

impl Reader for PipeStream {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        match self.obj.read(buf) {
            Ok(read) => Some(read),
            Err(ioerr) => {
                // EOF is indicated by returning None
                if ioerr.kind != EndOfFile {
                    read_error::cond.raise(ioerr);
                }
                return None;
            }
        }
    }

    fn eof(&mut self) -> bool { fail2!() }
}

impl Writer for PipeStream {
    fn write(&mut self, buf: &[u8]) {
        match self.obj.write(buf) {
            Ok(_) => (),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
            }
        }
    }

    fn flush(&mut self) { fail2!() }
}
