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
use cell::Cell;

pub trait Local {
    fn put(value: ~Self);
    fn take() -> ~Self;
    fn exists() -> bool;
    fn borrow<T>(f: &fn(&mut Self) -> T) -> T;
    unsafe fn unsafe_borrow() -> *mut Self;
    unsafe fn try_unsafe_borrow() -> Option<*mut Self>;
}

impl Local for Task {
    fn put(value: ~Task) { unsafe { local_ptr::put(value) } }
    fn take() -> ~Task { unsafe { local_ptr::take() } }
    fn exists() -> bool { local_ptr::exists() }
    fn borrow<T>(f: &fn(&mut Task) -> T) -> T {
        let mut res: Option<T> = None;
        let res_ptr: *mut Option<T> = &mut res;
        unsafe {
            do local_ptr::borrow |task| {
                let result = f(task);
                *res_ptr = Some(result);
            }
        }
        match res {
            Some(r) => { r }
            None => { rtabort!("function failed in local_borrow") }
        }
    }
    unsafe fn unsafe_borrow() -> *mut Task { local_ptr::unsafe_borrow() }
    unsafe fn try_unsafe_borrow() -> Option<*mut Task> {
        if Local::exists::<Task>() {
            Some(Local::unsafe_borrow())
        } else {
            None
        }
    }
}

impl Local for Scheduler {
    fn put(value: ~Scheduler) {
        let value = Cell::new(value);
        do Local::borrow::<Task,()> |task| {
            let task = task;
            task.sched = Some(value.take());
        };
    }
    fn take() -> ~Scheduler {
        do Local::borrow::<Task,~Scheduler> |task| {
            let sched = task.sched.take_unwrap();
            let task = task;
            task.sched = None;
            sched
        }
    }
    fn exists() -> bool {
        do Local::borrow::<Task,bool> |task| {
            match task.sched {
                Some(ref _task) => true,
                None => false
            }
        }
    }
    fn borrow<T>(f: &fn(&mut Scheduler) -> T) -> T {
        do Local::borrow::<Task, T> |task| {
            match task.sched {
                Some(~ref mut task) => {
                    f(task)
                }
                None => {
                    rtabort!("no scheduler")
                }
            }
        }
    }
    unsafe fn unsafe_borrow() -> *mut Scheduler {
        match (*Local::unsafe_borrow::<Task>()).sched {
            Some(~ref mut sched) => {
                let s: *mut Scheduler = &mut *sched;
                return s;
            }
            None => {
                rtabort!("no scheduler")
            }
        }
    }
    unsafe fn try_unsafe_borrow() -> Option<*mut Scheduler> {
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
    use option::None;
    use unstable::run_in_bare_thread;
    use rt::test::*;
    use super::*;
    use rt::task::Task;
    use rt::local_ptr;

    #[test]
    fn thread_local_task_smoke_test() {
        do run_in_bare_thread {
            local_ptr::init_tls_key();
            let mut sched = ~new_test_uv_sched();
            let task = ~Task::new_root(&mut sched.stack_pool, None, || {});
            Local::put(task);
            let task: ~Task = Local::take();
            cleanup_task(task);
        }
    }

    #[test]
    fn thread_local_task_two_instances() {
        do run_in_bare_thread {
            local_ptr::init_tls_key();
            let mut sched = ~new_test_uv_sched();
            let task = ~Task::new_root(&mut sched.stack_pool, None, || {});
            Local::put(task);
            let task: ~Task = Local::take();
            cleanup_task(task);
            let task = ~Task::new_root(&mut sched.stack_pool, None, || {});
            Local::put(task);
            let task: ~Task = Local::take();
            cleanup_task(task);
        }

    }

    #[test]
    fn borrow_smoke_test() {
        do run_in_bare_thread {
            local_ptr::init_tls_key();
            let mut sched = ~new_test_uv_sched();
            let task = ~Task::new_root(&mut sched.stack_pool, None, || {});
            Local::put(task);

            unsafe {
                let _task: *mut Task = Local::unsafe_borrow();
            }
            let task: ~Task = Local::take();
            cleanup_task(task);
        }
    }

    #[test]
    fn borrow_with_return() {
        do run_in_bare_thread {
            local_ptr::init_tls_key();
            let mut sched = ~new_test_uv_sched();
            let task = ~Task::new_root(&mut sched.stack_pool, None, || {});
            Local::put(task);

            let res = do Local::borrow::<Task,bool> |_task| {
                true
            };
            assert!(res)
                let task: ~Task = Local::take();
            cleanup_task(task);
        }
    }

}

