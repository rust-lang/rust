// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use option::{Option, Some, None};
use rt::sched::Scheduler;
use rt::task::Task;
use rt::local_ptr;
use rt::rtio::{EventLoop, IoFactoryObject};
//use borrow::to_uint;

pub trait Local {
    fn put(value: ~Self);
    fn take() -> ~Self;
    fn exists() -> bool;
    fn borrow<T>(f: &fn(&mut Self) -> T) -> T;
    unsafe fn unsafe_borrow() -> *mut Self;
    unsafe fn try_unsafe_borrow() -> Option<*mut Self>;
}

impl Local for Scheduler {
    fn put(value: ~Scheduler) { unsafe { local_ptr::put(value) }}
    fn take() -> ~Scheduler { unsafe { local_ptr::take() } }
    fn exists() -> bool { local_ptr::exists() }
    fn borrow<T>(f: &fn(&mut Scheduler) -> T) -> T {
        let mut res: Option<T> = None;
        let res_ptr: *mut Option<T> = &mut res;
        unsafe {
            do local_ptr::borrow |sched| {
//                rtdebug!("successfully unsafe borrowed sched pointer");
                let result = f(sched);
                *res_ptr = Some(result);
            }
        }
        match res {
            Some(r) => { r }
            None => rtabort!("function failed!")
        }
    }
    unsafe fn unsafe_borrow() -> *mut Scheduler { local_ptr::unsafe_borrow() }
    unsafe fn try_unsafe_borrow() -> Option<*mut Scheduler> { rtabort!("unimpl") }
}

impl Local for Task {
    fn put(_value: ~Task) { rtabort!("unimpl") }
    fn take() -> ~Task { rtabort!("unimpl") }
    fn exists() -> bool { rtabort!("unimpl") }
    fn borrow<T>(f: &fn(&mut Task) -> T) -> T {
        do Local::borrow::<Scheduler, T> |sched| {
//            rtdebug!("sched about to grab current_task");
            match sched.current_task {
                Some(~ref mut task) => {
//                    rtdebug!("current task pointer: %x", to_uint(task));
//                    rtdebug!("current task heap pointer: %x", to_uint(&task.heap));
                    f(task)
                }
                None => {
                    rtabort!("no scheduler")
                }
            }
        }
    }
    unsafe fn unsafe_borrow() -> *mut Task {
        match (*Local::unsafe_borrow::<Scheduler>()).current_task {
            Some(~ref mut task) => {
                let s: *mut Task = &mut *task;
                return s;
            }
            None => {
                // Don't fail. Infinite recursion
                rtabort!("no scheduler")
            }
        }
    }
    unsafe fn try_unsafe_borrow() -> Option<*mut Task> {
        if Local::exists::<Scheduler>() {
            Some(Local::unsafe_borrow())
        } else {
            None
        }
    }
}

// XXX: This formulation won't work once ~IoFactoryObject is a real trait pointer
impl Local for IoFactoryObject {
    fn put(_value: ~IoFactoryObject) { rtabort!("unimpl") }
    fn take() -> ~IoFactoryObject { rtabort!("unimpl") }
    fn exists() -> bool { rtabort!("unimpl") }
    fn borrow<T>(_f: &fn(&mut IoFactoryObject) -> T) -> T { rtabort!("unimpl") }
    unsafe fn unsafe_borrow() -> *mut IoFactoryObject {
        let sched = Local::unsafe_borrow::<Scheduler>();
        let io: *mut IoFactoryObject = (*sched).event_loop.io().unwrap();
        return io;
    }
    unsafe fn try_unsafe_borrow() -> Option<*mut IoFactoryObject> { rtabort!("unimpl") }
}

#[cfg(test)]
mod test {
    use unstable::run_in_bare_thread;
    use rt::test::*;
    use rt::sched::Scheduler;
    use super::*;

    #[test]
    fn thread_local_scheduler_smoke_test() {
        do run_in_bare_thread {
            let scheduler = ~new_test_uv_sched();
            Local::put(scheduler);
            let _scheduler: ~Scheduler = Local::take();
        }
    }

    #[test]
    fn thread_local_scheduler_two_instances() {
        do run_in_bare_thread {
            let scheduler = ~new_test_uv_sched();
            Local::put(scheduler);
            let _scheduler: ~Scheduler = Local::take();
            let scheduler = ~new_test_uv_sched();
            Local::put(scheduler);
            let _scheduler: ~Scheduler = Local::take();
        }
    }

    #[test]
    fn borrow_smoke_test() {
        do run_in_bare_thread {
            let scheduler = ~new_test_uv_sched();
            Local::put(scheduler);
            unsafe {
                let _scheduler: *mut Scheduler = Local::unsafe_borrow();
            }
            let _scheduler: ~Scheduler = Local::take();
        }
    }

    #[test]
    fn borrow_with_return() {
        do run_in_bare_thread {
            let scheduler = ~new_test_uv_sched();
            Local::put(scheduler);
            let res = do Local::borrow::<Scheduler,bool> |_sched| {
                true
            };
            assert!(res);
            let _scheduler: ~Scheduler = Local::take();
        }
    }

}
