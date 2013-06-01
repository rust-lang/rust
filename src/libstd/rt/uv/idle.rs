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
use option::Some;
use rt::uv::uvll;
use rt::uv::{Watcher, Loop, NativeHandle, IdleCallback, NullCallback};
use rt::uv::status_to_maybe_uv_error;

pub struct IdleWatcher(*uvll::uv_idle_t);
impl Watcher for IdleWatcher { }

impl IdleWatcher {
    pub fn new(loop_: &mut Loop) -> IdleWatcher {
        unsafe {
            let handle = uvll::idle_new();
            assert!(handle.is_not_null());
            assert!(0 == uvll::idle_init(loop_.native_handle(), handle));
            let mut watcher: IdleWatcher = NativeHandle::from_native_handle(handle);
            watcher.install_watcher_data();
            return watcher
        }
    }

    pub fn start(&mut self, cb: IdleCallback) {
        {
            let data = self.get_watcher_data();
            data.idle_cb = Some(cb);
        }

        unsafe {
            assert!(0 == uvll::idle_start(self.native_handle(), idle_cb))
        };

        extern fn idle_cb(handle: *uvll::uv_idle_t, status: c_int) {
            let mut idle_watcher: IdleWatcher = NativeHandle::from_native_handle(handle);
            let data = idle_watcher.get_watcher_data();
            let cb: &IdleCallback = data.idle_cb.get_ref();
            let status = status_to_maybe_uv_error(handle, status);
            (*cb)(idle_watcher, status);
        }
    }

    pub fn stop(&mut self) {
        // NB: Not resetting the Rust idle_cb to None here because `stop` is
        // likely called from *within* the idle callback, causing a use after
        // free

        unsafe {
            assert!(0 == uvll::idle_stop(self.native_handle()));
        }
    }

    pub fn close(self, cb: NullCallback) {
        {
            let mut this = self;
            let data = this.get_watcher_data();
            assert!(data.close_cb.is_none());
            data.close_cb = Some(cb);
        }

        unsafe { uvll::close(self.native_handle(), close_cb) };

        extern fn close_cb(handle: *uvll::uv_idle_t) {
            unsafe {
                let mut idle_watcher: IdleWatcher = NativeHandle::from_native_handle(handle);
                {
                    let data = idle_watcher.get_watcher_data();
                    data.close_cb.swap_unwrap()();
                }
                idle_watcher.drop_watcher_data();
                uvll::idle_delete(handle);
            }
        }
    }
}

impl NativeHandle<*uvll::uv_idle_t> for IdleWatcher {
    fn from_native_handle(handle: *uvll::uv_idle_t) -> IdleWatcher {
        IdleWatcher(handle)
    }
    fn native_handle(&self) -> *uvll::uv_idle_t {
        match self { &IdleWatcher(ptr) => ptr }
    }
}
