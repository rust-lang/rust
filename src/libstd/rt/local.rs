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

pub trait Local {
    fn put(value: ~Self);
    fn take() -> ~Self;
    fn exists() -> bool;
    fn borrow(f: &fn(&mut Self));
    unsafe fn unsafe_borrow() -> *mut Self;
    unsafe fn try_unsafe_borrow() -> Option<*mut Self>;
}

impl Local for Scheduler {
    fn put(value: ~Scheduler) { unsafe { local_ptr::put(value) }}
    fn take() -> ~Scheduler { unsafe { local_ptr::take() } }
    fn exists() -> bool { local_ptr::exists() }
    fn borrow(f: &fn(&mut Scheduler)) { unsafe { local_ptr::borrow(f) } }
    unsafe fn unsafe_borrow() -> *mut Scheduler { local_ptr::unsafe_borrow() }
    unsafe fn try_unsafe_borrow() -> Option<*mut Scheduler> { abort!("unimpl") }
}

impl Local for Task {
    fn put(_value: ~Task) { abort!("unimpl") }
    fn take() -> ~Task { abort!("unimpl") }
    fn exists() -> bool { abort!("unimpl") }
    fn borrow(f: &fn(&mut Task)) {
        do Local::borrow::<Scheduler> |sched| {
            match sched.current_task {
                Some(~ref mut task) => {
                    f(&mut *task.task)
                }
                None => {
                    abort!("no scheduler")
                }
            }
        }
    }
    unsafe fn unsafe_borrow() -> *mut Task {
        match (*Local::unsafe_borrow::<Scheduler>()).current_task {
            Some(~ref mut task) => {
                let s: *mut Task = &mut *task.task;
                return s;
            }
            None => {
                // Don't fail. Infinite recursion
                abort!("no scheduler")
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
    fn put(_value: ~IoFactoryObject) { abort!("unimpl") }
    fn take() -> ~IoFactoryObject { abort!("unimpl") }
    fn exists() -> bool { abort!("unimpl") }
    fn borrow(_f: &fn(&mut IoFactoryObject)) { abort!("unimpl") }
    unsafe fn unsafe_borrow() -> *mut IoFactoryObject {
        let sched = Local::unsafe_borrow::<Scheduler>();
        let io: *mut IoFactoryObject = (*sched).event_loop.io().unwrap();
        return io;
    }
    unsafe fn try_unsafe_borrow() -> Option<*mut IoFactoryObject> { abort!("unimpl") }
}

#[cfg(test)]
mod test {
    use rt::sched::Scheduler;
    use rt::uv::uvio::UvEventLoop;
    use super::*;

    #[test]
    fn thread_local_scheduler_smoke_test() {
        let scheduler = ~UvEventLoop::new_scheduler();
        Local::put(scheduler);
        let _scheduler: ~Scheduler = Local::take();
    }

    #[test]
    fn thread_local_scheduler_two_instances() {
        let scheduler = ~UvEventLoop::new_scheduler();
        Local::put(scheduler);
        let _scheduler: ~Scheduler = Local::take();
        let scheduler = ~UvEventLoop::new_scheduler();
        Local::put(scheduler);
        let _scheduler: ~Scheduler = Local::take();
    }

    #[test]
    fn borrow_smoke_test() {
        let scheduler = ~UvEventLoop::new_scheduler();
        Local::put(scheduler);
        unsafe {
            let _scheduler: *mut Scheduler = Local::unsafe_borrow();
        }
        let _scheduler: ~Scheduler = Local::take();
    }
}
