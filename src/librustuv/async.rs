// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::libc::c_int;

use uvll;
use super::{Watcher, Loop, NativeHandle, AsyncCallback, WatcherInterop};
use super::status_to_maybe_uv_error;

pub struct AsyncWatcher(*uvll::uv_async_t);
impl Watcher for AsyncWatcher { }

impl AsyncWatcher {
    pub fn new(loop_: &mut Loop, cb: AsyncCallback) -> AsyncWatcher {
        unsafe {
            let handle = uvll::malloc_handle(uvll::UV_ASYNC);
            assert!(handle.is_not_null());
            let mut watcher: AsyncWatcher = NativeHandle::from_native_handle(handle);
            watcher.install_watcher_data();
            let data = watcher.get_watcher_data();
            data.async_cb = Some(cb);
            assert_eq!(0, uvll::async_init(loop_.native_handle(), handle, async_cb));
            return watcher;
        }

        extern fn async_cb(handle: *uvll::uv_async_t, status: c_int) {
            let mut watcher: AsyncWatcher = NativeHandle::from_native_handle(handle);
            let status = status_to_maybe_uv_error(status);
            let data = watcher.get_watcher_data();
            let cb = data.async_cb.get_ref();
            (*cb)(watcher, status);
        }
    }

    pub fn send(&mut self) {
        unsafe {
            let handle = self.native_handle();
            uvll::async_send(handle);
        }
    }
}

impl NativeHandle<*uvll::uv_async_t> for AsyncWatcher {
    fn from_native_handle(handle: *uvll::uv_async_t) -> AsyncWatcher {
        AsyncWatcher(handle)
    }
    fn native_handle(&self) -> *uvll::uv_async_t {
        match self { &AsyncWatcher(ptr) => ptr }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use Loop;
    use std::unstable::run_in_bare_thread;
    use std::rt::thread::Thread;
    use std::cell::Cell;

    #[test]
    fn smoke_test() {
        do run_in_bare_thread {
            let mut loop_ = Loop::new();
            let watcher = AsyncWatcher::new(&mut loop_, |w, _| w.close(||()) );
            let watcher_cell = Cell::new(watcher);
            let thread = do Thread::start {
                let mut watcher = watcher_cell.take();
                watcher.send();
            };
            loop_.run();
            loop_.close();
            thread.join();
        }
    }
}
