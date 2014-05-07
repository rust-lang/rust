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
use libc::c_void;

use uvll;
use super::{Loop, UvHandle};
use std::rt::rtio::{Callback, PausableIdleCallback};

pub struct IdleWatcher {
    handle: *uvll::uv_idle_t,
    idle_flag: bool,
    closed: bool,
    callback: Box<Callback:Send>,
}

impl IdleWatcher {
    pub fn new(loop_: &mut Loop, cb: Box<Callback:Send>) -> Box<IdleWatcher> {
        let handle = UvHandle::alloc(None::<IdleWatcher>, uvll::UV_IDLE);
        assert_eq!(unsafe {
            uvll::uv_idle_init(loop_.handle, handle)
        }, 0);
        let me = box IdleWatcher {
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
            let data: *c_void = cast::transmute(box f);
            uvll::set_data_for_uv_handle(handle, data);
            assert_eq!(uvll::uv_idle_start(handle, onetime_cb), 0)
        }

        extern fn onetime_cb(handle: *uvll::uv_idle_t) {
            unsafe {
                let data = uvll::get_data_for_uv_handle(handle);
                let f: Box<proc()> = cast::transmute(data);
                (*f)();
                assert_eq!(uvll::uv_idle_stop(handle), 0);
                uvll::uv_close(handle, close_cb);
            }
        }

        extern fn close_cb(handle: *uvll::uv_handle_t) {
            unsafe { uvll::free_handle(handle) }
        }
    }
}

impl PausableIdleCallback for IdleWatcher {
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

extern fn idle_cb(handle: *uvll::uv_idle_t) {
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
    use std::cast;
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::rt::rtio::{Callback, PausableIdleCallback};
    use std::rt::task::{BlockedTask, Task};
    use std::rt::local::Local;
    use super::IdleWatcher;
    use super::super::local_loop;

    type Chan = Rc<RefCell<(Option<BlockedTask>, uint)>>;

    struct MyCallback(Rc<RefCell<(Option<BlockedTask>, uint)>>, uint);
    impl Callback for MyCallback {
        fn call(&mut self) {
            let task = match *self {
                MyCallback(ref rc, n) => {
                    match *rc.borrow_mut().deref_mut() {
                        (ref mut task, ref mut val) => {
                            *val = n;
                            match task.take() {
                                Some(t) => t,
                                None => return
                            }
                        }
                    }
                }
            };
            let _ = task.wake().map(|t| t.reawaken());
        }
    }

    fn mk(v: uint) -> (Box<IdleWatcher>, Chan) {
        let rc = Rc::new(RefCell::new((None, 0)));
        let cb = box MyCallback(rc.clone(), v);
        let cb = cb as Box<Callback:>;
        let cb = unsafe { cast::transmute(cb) };
        (IdleWatcher::new(&mut local_loop().loop_, cb), rc)
    }

    fn sleep(chan: &Chan) -> uint {
        let task: Box<Task> = Local::take();
        task.deschedule(1, |task| {
            match *chan.borrow_mut().deref_mut() {
                (ref mut slot, _) => {
                    assert!(slot.is_none());
                    *slot = Some(task);
                }
            }
            Ok(())
        });

        match *chan.borrow() { (_, n) => n }
    }

    #[test]
    fn not_used() {
        let (_idle, _chan) = mk(1);
    }

    #[test]
    fn smoke_test() {
        let (mut idle, chan) = mk(1);
        idle.resume();
        assert_eq!(sleep(&chan), 1);
    }

    #[test] #[should_fail]
    fn smoke_fail() {
        // By default, the test harness is capturing our stderr output through a
        // channel. This means that when we start failing and "print" our error
        // message, we could be switched to running on another test. The
        // IdleWatcher assumes that we're already running on the same task, so
        // it can cause serious problems and internal race conditions.
        //
        // To fix this bug, we just set our stderr to a null writer which will
        // never reschedule us, so we're guaranteed to stay on the same
        // task/event loop.
        use std::io;
        drop(io::stdio::set_stderr(box io::util::NullWriter));

        let (mut idle, _chan) = mk(1);
        idle.resume();
        fail!();
    }

    #[test]
    fn fun_combinations_of_methods() {
        let (mut idle, chan) = mk(1);
        idle.resume();
        assert_eq!(sleep(&chan), 1);
        idle.pause();
        idle.resume();
        idle.resume();
        assert_eq!(sleep(&chan), 1);
        idle.pause();
        idle.pause();
        idle.resume();
        assert_eq!(sleep(&chan), 1);
    }

    #[test]
    fn pause_pauses() {
        let (mut idle1, chan1) = mk(1);
        let (mut idle2, chan2) = mk(2);
        idle2.resume();
        assert_eq!(sleep(&chan2), 2);
        idle2.pause();
        idle1.resume();
        assert_eq!(sleep(&chan1), 1);
    }
}
