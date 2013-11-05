// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cast;
use std::libc::{c_int, c_void};

use uvll;
use super::{Loop, UvHandle};
use std::rt::rtio::{Callback, PausibleIdleCallback};

pub struct IdleWatcher {
    handle: *uvll::uv_idle_t,
    idle_flag: bool,
    closed: bool,
    callback: Option<~Callback>,
}

impl IdleWatcher {
    pub fn new(loop_: &mut Loop) -> ~IdleWatcher {
        let handle = UvHandle::alloc(None::<IdleWatcher>, uvll::UV_IDLE);
        assert_eq!(unsafe {
            uvll::uv_idle_init(loop_.handle, handle)
        }, 0);
        let me = ~IdleWatcher {
            handle: handle,
            idle_flag: false,
            closed: false,
            callback: None,
        };
        return me.install();
    }

    pub fn onetime(loop_: &mut Loop, f: proc()) {
        let handle = UvHandle::alloc(None::<IdleWatcher>, uvll::UV_IDLE);
        unsafe {
            assert_eq!(uvll::uv_idle_init(loop_.handle, handle), 0);
            let data: *c_void = cast::transmute(~f);
            uvll::set_data_for_uv_handle(handle, data);
            assert_eq!(uvll::uv_idle_start(handle, onetime_cb), 0)
        }

        extern fn onetime_cb(handle: *uvll::uv_idle_t, status: c_int) {
            assert_eq!(status, 0);
            unsafe {
                let data = uvll::get_data_for_uv_handle(handle);
                let f: ~proc() = cast::transmute(data);
                (*f)();
                uvll::uv_idle_stop(handle);
                uvll::uv_close(handle, close_cb);
            }
        }

        extern fn close_cb(handle: *uvll::uv_handle_t) {
            unsafe { uvll::free_handle(handle) }
        }
    }
}

impl PausibleIdleCallback for IdleWatcher {
    fn start(&mut self, cb: ~Callback) {
        assert!(self.callback.is_none());
        self.callback = Some(cb);
        assert_eq!(unsafe { uvll::uv_idle_start(self.handle, idle_cb) }, 0)
        self.idle_flag = true;
    }
    fn pause(&mut self) {
        if self.idle_flag == true {
            assert_eq!(unsafe {uvll::uv_idle_stop(self.handle) }, 0);
            self.idle_flag = false;
        }
    }
    fn resume(&mut self) {
        if self.idle_flag == false {
            assert_eq!(unsafe { uvll::uv_idle_start(self.handle, idle_cb) }, 0)
            self.idle_flag = true;
        }
    }
    fn close(&mut self) {
        self.pause();
        if !self.closed {
            self.closed = true;
            self.close_async_();
        }
    }
}

impl UvHandle<uvll::uv_idle_t> for IdleWatcher {
    fn uv_handle(&self) -> *uvll::uv_idle_t { self.handle }
}

extern fn idle_cb(handle: *uvll::uv_idle_t, status: c_int) {
    assert_eq!(status, 0);
    let idle: &mut IdleWatcher = unsafe { UvHandle::from_uv_handle(&handle) };
    assert!(idle.callback.is_some());
    idle.callback.get_mut_ref().call();
}

#[cfg(test)]
mod test {

    use Loop;
    use super::*;
    use std::unstable::run_in_bare_thread;

    #[test]
    #[ignore(reason = "valgrind - loop destroyed before watcher?")]
    fn idle_new_then_close() {
        do run_in_bare_thread {
            let mut loop_ = Loop::new();
            let idle_watcher = { IdleWatcher::new(&mut loop_) };
            idle_watcher.close(||());
        }
    }

    #[test]
    fn idle_smoke_test() {
        do run_in_bare_thread {
            let mut loop_ = Loop::new();
            let mut idle_watcher = { IdleWatcher::new(&mut loop_) };
            let mut count = 10;
            let count_ptr: *mut int = &mut count;
            do idle_watcher.start |idle_watcher, status| {
                let mut idle_watcher = idle_watcher;
                assert!(status.is_none());
                if unsafe { *count_ptr == 10 } {
                    idle_watcher.stop();
                    idle_watcher.close(||());
                } else {
                    unsafe { *count_ptr = *count_ptr + 1; }
                }
            }
            loop_.run();
            loop_.close();
            assert_eq!(count, 10);
        }
    }

    #[test]
    fn idle_start_stop_start() {
        do run_in_bare_thread {
            let mut loop_ = Loop::new();
            let mut idle_watcher = { IdleWatcher::new(&mut loop_) };
            do idle_watcher.start |idle_watcher, status| {
                let mut idle_watcher = idle_watcher;
                assert!(status.is_none());
                idle_watcher.stop();
                do idle_watcher.start |idle_watcher, status| {
                    assert!(status.is_none());
                    let mut idle_watcher = idle_watcher;
                    idle_watcher.stop();
                    idle_watcher.close(||());
                }
            }
            loop_.run();
            loop_.close();
        }
    }
}
