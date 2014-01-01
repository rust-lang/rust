// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use option::Option;
use rt::task::Task;
use rt::local_ptr;

/// Encapsulates some task-local data.
pub trait Local<Borrowed> {
    fn put(value: ~Self);
    fn take() -> ~Self;
    fn try_take() -> Option<~Self>;
    fn exists(unused_value: Option<Self>) -> bool;
    fn borrow(unused_value: Option<Self>) -> Borrowed;
    unsafe fn unsafe_take() -> ~Self;
    unsafe fn unsafe_borrow() -> *mut Self;
    unsafe fn try_unsafe_borrow() -> Option<*mut Self>;
}

impl Local<local_ptr::Borrowed<Task>> for Task {
    #[inline]
    fn put(value: ~Task) { unsafe { local_ptr::put(value) } }
    #[inline]
    fn take() -> ~Task { unsafe { local_ptr::take() } }
    #[inline]
    fn try_take() -> Option<~Task> { unsafe { local_ptr::try_take() } }
    fn exists(_: Option<Task>) -> bool { local_ptr::exists() }
    #[inline]
    fn borrow(_: Option<Task>) -> local_ptr::Borrowed<Task> {
        unsafe {
            local_ptr::borrow::<Task>()
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

#[cfg(test)]
mod test {
    use option::{None, Option};
    use unstable::run_in_bare_thread;
    use super::*;
    use rt::task::Task;
    use rt::local_ptr;

    #[test]
    fn thread_local_task_smoke_test() {
        do run_in_bare_thread {
            let task = ~Task::new();
            Local::put(task);
            let task: ~Task = Local::take();
            cleanup_task(task);
        }
    }

    #[test]
    fn thread_local_task_two_instances() {
        do run_in_bare_thread {
            let task = ~Task::new();
            Local::put(task);
            let task: ~Task = Local::take();
            cleanup_task(task);
            let task = ~Task::new();
            Local::put(task);
            let task: ~Task = Local::take();
            cleanup_task(task);
        }

    }

    #[test]
    fn borrow_smoke_test() {
        do run_in_bare_thread {
            let task = ~Task::new();
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
            let task = ~Task::new();
            Local::put(task);

            {
                let _ = Local::borrow(None::<Task>);
            }

            let task: ~Task = Local::take();
            cleanup_task(task);
        }
    }

    #[test]
    fn try_take() {
        do run_in_bare_thread {
            let task = ~Task::new();
            Local::put(task);

            let t: ~Task = Local::try_take().unwrap();
            let u: Option<~Task> = Local::try_take();
            assert!(u.is_none());

            cleanup_task(t);
        }
    }

    fn cleanup_task(mut t: ~Task) {
        t.destroyed = true;
    }

}

