// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc::c_int;
use std::rt::rtio::{RtioSignal, Callback};

use homing::{HomingIO, HomeHandle};
use super::{UvError, UvHandle};
use uvll;
use uvio::UvIoFactory;

pub struct SignalWatcher {
    handle: *uvll::uv_signal_t,
    home: HomeHandle,

    cb: Box<Callback:Send>,
}

impl SignalWatcher {
    pub fn new(io: &mut UvIoFactory, signum: int, cb: Box<Callback:Send>)
               -> Result<Box<SignalWatcher>, UvError> {
        let s = box SignalWatcher {
            handle: UvHandle::alloc(None::<SignalWatcher>, uvll::UV_SIGNAL),
            home: io.make_handle(),
            cb: cb,
        };
        assert_eq!(unsafe {
            uvll::uv_signal_init(io.uv_loop(), s.handle)
        }, 0);

        match unsafe {
            uvll::uv_signal_start(s.handle, signal_cb, signum as c_int)
        } {
            0 => Ok(s.install()),
            n => Err(UvError(n)),
        }

    }
}

extern fn signal_cb(handle: *uvll::uv_signal_t, _signum: c_int) {
    let s: &mut SignalWatcher = unsafe { UvHandle::from_uv_handle(&handle) };
    let _ = s.cb.call();
}

impl HomingIO for SignalWatcher {
    fn home<'r>(&'r mut self) -> &'r mut HomeHandle { &mut self.home }
}

impl UvHandle<uvll::uv_signal_t> for SignalWatcher {
    fn uv_handle(&self) -> *uvll::uv_signal_t { self.handle }
}

impl RtioSignal for SignalWatcher {}

impl Drop for SignalWatcher {
    fn drop(&mut self) {
        let _m = self.fire_homing_missile();
        self.close();
    }
}
