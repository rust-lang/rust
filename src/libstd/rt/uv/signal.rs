// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cast;
use option::Some;
use libc::c_int;
use result::{Err, Ok, Result};
use rt::io::signal::Signum;
use rt::uv::{Loop, NativeHandle, SignalCallback, UvError, Watcher};
use rt::uv::uvll;

pub struct SignalWatcher(*uvll::uv_signal_t);

impl Watcher for SignalWatcher { }

impl SignalWatcher {
    pub fn new(loop_: &mut Loop) -> SignalWatcher {
        unsafe {
            let handle = uvll::malloc_handle(uvll::UV_SIGNAL);
            assert!(handle.is_not_null());
            assert!(0 == uvll::signal_init(loop_.native_handle(), handle));
            let mut watcher: SignalWatcher = NativeHandle::from_native_handle(handle);
            watcher.install_watcher_data();
            return watcher;
        }
    }

    pub fn start(&mut self, signum: Signum, callback: SignalCallback)
            -> Result<(), UvError>
    {
        return unsafe {
            match uvll::signal_start(self.native_handle(), signal_cb,
                                     signum as c_int) {
                0 => {
                    let data = self.get_watcher_data();
                    data.signal_cb = Some(callback);
                    Ok(())
                }
                n => Err(UvError(n)),
            }
        };

        extern fn signal_cb(handle: *uvll::uv_signal_t, signum: c_int) {
            let mut watcher: SignalWatcher = NativeHandle::from_native_handle(handle);
            let data = watcher.get_watcher_data();
            let cb = data.signal_cb.get_ref();
            (*cb)(watcher, unsafe { cast::transmute(signum as int) });
        }
    }

    pub fn stop(&mut self) {
        unsafe {
            uvll::signal_stop(self.native_handle());
        }
    }
}

impl NativeHandle<*uvll::uv_signal_t> for SignalWatcher {
    fn from_native_handle(handle: *uvll::uv_signal_t) -> SignalWatcher {
        SignalWatcher(handle)
    }

    fn native_handle(&self) -> *uvll::uv_signal_t {
        match self { &SignalWatcher(ptr) => ptr }
    }
}
