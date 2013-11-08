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
    callback: ~Callback,
}

impl IdleWatcher {
    pub fn new(loop_: &mut Loop, cb: ~Callback) -> ~IdleWatcher {
        let handle = UvHandle::alloc(None::<IdleWatcher>, uvll::UV_IDLE);
        assert_eq!(unsafe {
            uvll::uv_idle_init(loop_.handle, handle)
        }, 0);
        let me = ~IdleWatcher {
            handle: handle,
            idle_flag: false,
            closed: false,
            callback: cb,
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
}

impl UvHandle<uvll::uv_idle_t> for IdleWatcher {
    fn uv_handle(&self) -> *uvll::uv_idle_t { self.handle }
}

extern fn idle_cb(handle: *uvll::uv_idle_t, status: c_int) {
    assert_eq!(status, 0);
    let idle: &mut IdleWatcher = unsafe { UvHandle::from_uv_handle(&handle) };
    idle.callback.call();
}

impl Drop for IdleWatcher {
    fn drop(&mut self) {
        self.pause();
        self.close_async_();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::rt::tube::Tube;
    use std::rt::rtio::{Callback, PausibleIdleCallback};
    use super::super::local_loop;

    struct MyCallback(Tube<int>, int);
    impl Callback for MyCallback {
        fn call(&mut self) {
            match *self {
                MyCallback(ref mut tube, val) => tube.send(val)
            }
        }
    }

    #[test]
    fn not_used() {
        let cb = ~MyCallback(Tube::new(), 1);
        let _idle = IdleWatcher::new(local_loop(), cb as ~Callback);
    }

    #[test]
    fn smoke_test() {
        let mut tube = Tube::new();
        let cb = ~MyCallback(tube.clone(), 1);
        let mut idle = IdleWatcher::new(local_loop(), cb as ~Callback);
        idle.resume();
        tube.recv();
    }

    #[test] #[should_fail]
    fn smoke_fail() {
        let tube = Tube::new();
        let cb = ~MyCallback(tube.clone(), 1);
        let mut idle = IdleWatcher::new(local_loop(), cb as ~Callback);
        idle.resume();
        fail!();
    }

    #[test]
    fn fun_combinations_of_methods() {
        let mut tube = Tube::new();
        let cb = ~MyCallback(tube.clone(), 1);
        let mut idle = IdleWatcher::new(local_loop(), cb as ~Callback);
        idle.resume();
        tube.recv();
        idle.pause();
        idle.resume();
        idle.resume();
        tube.recv();
        idle.pause();
        idle.pause();
        idle.resume();
        tube.recv();
    }

    #[test]
    fn pause_pauses() {
        let mut tube = Tube::new();
        let cb = ~MyCallback(tube.clone(), 1);
        let mut idle1 = IdleWatcher::new(local_loop(), cb as ~Callback);
        let cb = ~MyCallback(tube.clone(), 2);
        let mut idle2 = IdleWatcher::new(local_loop(), cb as ~Callback);
        idle2.resume();
        assert_eq!(tube.recv(), 2);
        idle2.pause();
        idle1.resume();
        assert_eq!(tube.recv(), 1);
    }
}
