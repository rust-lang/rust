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

pub struct PipeStream(RtioPipeObject);
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
        PipeStream(inner)
    }
}

impl Reader for PipeStream {
    fn read(&mut self, buf: &mut [u8]) -> Option<uint> {
        match (**self).read(buf) {
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

    fn eof(&mut self) -> bool { fail!() }
}

impl Writer for PipeStream {
    fn write(&mut self, buf: &[u8]) {
        match (**self).write(buf) {
            Ok(_) => (),
            Err(ioerr) => {
                io_error::cond.raise(ioerr);
            }
        }
    }

    fn flush(&mut self) { fail!() }
}
