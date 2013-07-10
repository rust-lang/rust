// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc::{c_int, c_void};
use option::Some;
use rt::uv::uvll;
use rt::uv::uvll::UV_ASYNC;
use rt::uv::{Watcher, Loop, NativeHandle, AsyncCallback, NullCallback};
use rt::uv::WatcherInterop;
use rt::uv::status_to_maybe_uv_error;

pub struct AsyncWatcher(*uvll::uv_async_t);
impl Watcher for AsyncWatcher { }

impl AsyncWatcher {
    pub fn new(loop_: &mut Loop, cb: AsyncCallback) -> AsyncWatcher {
        unsafe {
            let handle = uvll::malloc_handle(UV_ASYNC);
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
            let status = status_to_maybe_uv_error(watcher.native_handle(), status);
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

    pub fn close(self, cb: NullCallback) {
        let mut this = self;
        let data = this.get_watcher_data();
        assert!(data.close_cb.is_none());
        data.close_cb = Some(cb);

        unsafe {
            uvll::close(self.native_handle(), close_cb);
        }

        extern fn close_cb(handle: *uvll::uv_stream_t) {
            let mut watcher: AsyncWatcher = NativeHandle::from_native_handle(handle);
            {
                let data = watcher.get_watcher_data();
                data.close_cb.swap_unwrap()();
            }
            watcher.drop_watcher_data();
            unsafe { uvll::free_handle(handle as *c_void); }
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
    use rt::uv::Loop;
    use unstable::run_in_bare_thread;
    use rt::thread::Thread;
    use cell::Cell;

    #[test]
    fn smoke_test() {
        do run_in_bare_thread {
            let mut loop_ = Loop::new();
            let watcher = AsyncWatcher::new(&mut loop_, |w, _| w.close(||()) );
            let watcher_cell = Cell::new(watcher);
            let _thread = do Thread::start {
                let mut watcher = watcher_cell.take();
                watcher.send();
            };
            loop_.run();
            loop_.close();
        }
    }
}
