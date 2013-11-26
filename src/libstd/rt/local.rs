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
use cell::Cell;

pub trait Local {
    fn put(value: ~Self);
    fn take() -> ~Self;
    fn exists(unused_value: Option<Self>) -> bool;
    fn borrow<T>(f: |&mut Self| -> T) -> T;
    unsafe fn unsafe_take() -> ~Self;
    unsafe fn unsafe_borrow() -> *mut Self;
    unsafe fn try_unsafe_borrow() -> Option<*mut Self>;
}

impl Local for Task {
    #[inline]
    fn put(value: ~Task) { unsafe { local_ptr::put(value) } }
    #[inline]
    fn take() -> ~Task { unsafe { local_ptr::take() } }
    fn exists(_: Option<Task>) -> bool { local_ptr::exists() }
    fn borrow<T>(f: |&mut Task| -> T) -> T {
        let mut res: Option<T> = None;
        let res_ptr: *mut Option<T> = &mut res;
        unsafe {
            local_ptr::borrow(|task| {
                let result = f(task);
                *res_ptr = Some(result);
            })
        }
        match res {
            Some(r) => { r }
            None => { rtabort!("function failed in local_borrow") }
        }
    }
    #[inline]
    unsafe fn unsafe_take() -> ~Task { local_ptr::unsafe_take() }
    #[inline]
    unsafe fn unsafe_borrow() -> *mut Task { local_ptr::unsafe_borrow() }
    #[inline]
    unsafe fn try_unsafe_borrow() -> Option<*mut Task> {
        local_ptr::try_unsafe_borrow()
    }
}

impl Local for Scheduler {
    fn put(value: ~Scheduler) {
        let value = Cell::new(value);
        Local::borrow(|task: &mut Task| {
            let task = task;
            task.sched = Some(value.take());
        });
    }
    #[inline]
    fn take() -> ~Scheduler {
        unsafe {
            // XXX: Unsafe for speed
            let task: *mut Task = Local::unsafe_borrow();
            (*task).sched.take_unwrap()
        }
    }
    fn exists(_: Option<Scheduler>) -> bool {
        Local::borrow(|task: &mut Task| {
            match task.sched {
                Some(ref _task) => true,
                None => false
            }
        })
    }
    fn borrow<T>(f: |&mut Scheduler| -> T) -> T {
        Local::borrow(|task: &mut Task| {
            match task.sched {
                Some(~ref mut task) => {
                    f(task)
                }
                None => {
                    rtabort!("no scheduler")
                }
            }
        })
    }
    unsafe fn unsafe_take() -> ~Scheduler { rtabort!("unimpl") }
    unsafe fn unsafe_borrow() -> *mut Scheduler {
        let task: *mut Task = Local::unsafe_borrow();
        match (*task).sched {
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
        let task_opt: Option<*mut Task> = Local::try_unsafe_borrow();
        match task_opt {
            Some(task) => {
                match (*task).sched {
                    Some(~ref mut sched) => {
                        let s: *mut Scheduler = &mut *sched;
                        Some(s)
                    }
                    None => None
                }
            }
            None => None
        }
    }
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
            let task = ~Task::new_root(&mut sched.stack_pool, None, proc(){});
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
            let task = ~Task::new_root(&mut sched.stack_pool, None, proc(){});
            Local::put(task);
            let task: ~Task = Local::take();
            cleanup_task(task);
            let task = ~Task::new_root(&mut sched.stack_pool, None, proc(){});
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
            let task = ~Task::new_root(&mut sched.stack_pool, None, proc(){});
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
            let task = ~Task::new_root(&mut sched.stack_pool, None, proc(){});
            Local::put(task);

            let res = Local::borrow(|_task: &mut Task| {
                true
            });
            assert!(res)
                let task: ~Task = Local::take();
            cleanup_task(task);
        }
    }

}

