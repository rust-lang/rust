// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The EventLoop and internal synchronous I/O interface.

use core::prelude::*;
use alloc::boxed::Box;
use collections::string::String;
use core::mem;
use libc::c_int;

use local::Local;
use task::Task;

pub trait EventLoop {
    fn run(&mut self);
    fn callback(&mut self, arg: proc(): Send);
    fn pausable_idle_callback(&mut self, Box<Callback + Send>)
                              -> Box<PausableIdleCallback + Send>;
    fn remote_callback(&mut self, Box<Callback + Send>)
                       -> Box<RemoteCallback + Send>;

    /// The asynchronous I/O services. Not all event loops may provide one.
    fn io<'a>(&'a mut self) -> Option<&'a mut IoFactory>;
    fn has_active_io(&self) -> bool;
}

pub trait Callback {
    fn call(&mut self);
}

pub trait RemoteCallback {
    /// Trigger the remote callback. Note that the number of times the
    /// callback is run is not guaranteed. All that is guaranteed is
    /// that, after calling 'fire', the callback will be called at
    /// least once, but multiple callbacks may be coalesced and
    /// callbacks may be called more often requested. Destruction also
    /// triggers the callback.
    fn fire(&mut self);
}

pub struct LocalIo<'a> {
    factory: &'a mut IoFactory+'a,
}

#[unsafe_destructor]
impl<'a> Drop for LocalIo<'a> {
    fn drop(&mut self) {
        // FIXME(pcwalton): Do nothing here for now, but eventually we may want
        // something. For now this serves to make `LocalIo` noncopyable.
    }
}

impl<'a> LocalIo<'a> {
    /// Returns the local I/O: either the local scheduler's I/O services or
    /// the native I/O services.
    pub fn borrow() -> Option<LocalIo<'a>> {
        // FIXME(#11053): bad
        //
        // This is currently very unsafely implemented. We don't actually
        // *take* the local I/O so there's a very real possibility that we
        // can have two borrows at once. Currently there is not a clear way
        // to actually borrow the local I/O factory safely because even if
        // ownership were transferred down to the functions that the I/O
        // factory implements it's just too much of a pain to know when to
        // relinquish ownership back into the local task (but that would be
        // the safe way of implementing this function).
        //
        // In order to get around this, we just transmute a copy out of the task
        // in order to have what is likely a static lifetime (bad).
        let mut t: Box<Task> = match Local::try_take() {
            Some(t) => t,
            None => return None,
        };
        let ret = t.local_io().map(|t| {
            unsafe { mem::transmute_copy(&t) }
        });
        Local::put(t);
        return ret;
    }

    pub fn maybe_raise<T>(f: |io: &mut IoFactory| -> IoResult<T>)
        -> IoResult<T>
    {
        #[cfg(unix)] use libc::EINVAL as ERROR;
        #[cfg(windows)] use libc::ERROR_CALL_NOT_IMPLEMENTED as ERROR;
        match LocalIo::borrow() {
            Some(mut io) => f(io.get()),
            None => Err(IoError {
                code: ERROR as uint,
                extra: 0,
                detail: None,
            }),
        }
    }

    pub fn new<'a>(io: &'a mut IoFactory+'a) -> LocalIo<'a> {
        LocalIo { factory: io }
    }

    /// Returns the underlying I/O factory as a trait reference.
    #[inline]
    pub fn get<'a>(&'a mut self) -> &'a mut IoFactory {
        let f: &'a mut IoFactory = self.factory;
        f
    }
}

pub trait IoFactory {
    fn timer_init(&mut self) -> IoResult<Box<RtioTimer + Send>>;
    fn tty_open(&mut self, fd: c_int, readable: bool)
            -> IoResult<Box<RtioTTY + Send>>;
}

pub trait RtioTimer {
    fn sleep(&mut self, msecs: u64);
    fn oneshot(&mut self, msecs: u64, cb: Box<Callback + Send>);
    fn period(&mut self, msecs: u64, cb: Box<Callback + Send>);
}

pub trait RtioPipe {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint>;
    fn write(&mut self, buf: &[u8]) -> IoResult<()>;
    fn clone(&self) -> Box<RtioPipe + Send>;

    fn close_write(&mut self) -> IoResult<()>;
    fn close_read(&mut self) -> IoResult<()>;
    fn set_timeout(&mut self, timeout_ms: Option<u64>);
    fn set_read_timeout(&mut self, timeout_ms: Option<u64>);
    fn set_write_timeout(&mut self, timeout_ms: Option<u64>);
}

pub trait RtioUnixListener {
    fn listen(self: Box<Self>) -> IoResult<Box<RtioUnixAcceptor + Send>>;
}

pub trait RtioUnixAcceptor {
    fn accept(&mut self) -> IoResult<Box<RtioPipe + Send>>;
    fn set_timeout(&mut self, timeout: Option<u64>);
    fn clone(&self) -> Box<RtioUnixAcceptor + Send>;
    fn close_accept(&mut self) -> IoResult<()>;
}

pub trait RtioTTY {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint>;
    fn write(&mut self, buf: &[u8]) -> IoResult<()>;
    fn set_raw(&mut self, raw: bool) -> IoResult<()>;
    fn get_winsize(&mut self) -> IoResult<(int, int)>;
    fn isatty(&self) -> bool;
}

pub trait PausableIdleCallback {
    fn pause(&mut self);
    fn resume(&mut self);
}

pub trait RtioSignal {}

#[deriving(Show)]
pub struct IoError {
    pub code: uint,
    pub extra: uint,
    pub detail: Option<String>,
}

pub type IoResult<T> = Result<T, IoError>;
