// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use alloc::owned::Box;
use local_ptr;
use task::Task;

/// Encapsulates some task-local data.
pub trait Local<Borrowed> {
    fn put(value: Box<Self>);
    fn take() -> Box<Self>;
    fn try_take() -> Option<Box<Self>>;
    fn exists(unused_value: Option<Self>) -> bool;
    fn borrow(unused_value: Option<Self>) -> Borrowed;
    unsafe fn unsafe_take() -> Box<Self>;
    unsafe fn unsafe_borrow() -> *mut Self;
    unsafe fn try_unsafe_borrow() -> Option<*mut Self>;
}

#[allow(visible_private_types)]
impl Local<local_ptr::Borrowed<Task>> for Task {
    #[inline]
    fn put(value: Box<Task>) { unsafe { local_ptr::put(value) } }
    #[inline]
    fn take() -> Box<Task> { unsafe { local_ptr::take() } }
    #[inline]
    fn try_take() -> Option<Box<Task>> { unsafe { local_ptr::try_take() } }
    fn exists(_: Option<Task>) -> bool { local_ptr::exists() }
    #[inline]
    fn borrow(_: Option<Task>) -> local_ptr::Borrowed<Task> {
        unsafe {
            local_ptr::borrow::<Task>()
        }
    }
    #[inline]
    unsafe fn unsafe_take() -> Box<Task> { local_ptr::unsafe_take() }
    #[inline]
    unsafe fn unsafe_borrow() -> *mut Task { local_ptr::unsafe_borrow() }
    #[inline]
    unsafe fn try_unsafe_borrow() -> Option<*mut Task> {
        local_ptr::try_unsafe_borrow()
    }
}

#[cfg(test)]
mod test {
    use std::prelude::*;
    use std::rt::thread::Thread;
    use super::*;
    use task::Task;

    #[test]
    fn thread_local_task_smoke_test() {
        Thread::start(proc() {
            let task = box Task::new();
            Local::put(task);
            let task: Box<Task> = Local::take();
            cleanup_task(task);
        }).join();
    }

    #[test]
    fn thread_local_task_two_instances() {
        Thread::start(proc() {
            let task = box Task::new();
            Local::put(task);
            let task: Box<Task> = Local::take();
            cleanup_task(task);
            let task = box Task::new();
            Local::put(task);
            let task: Box<Task> = Local::take();
            cleanup_task(task);
        }).join();
    }

    #[test]
    fn borrow_smoke_test() {
        Thread::start(proc() {
            let task = box Task::new();
            Local::put(task);

            unsafe {
                let _task: *mut Task = Local::unsafe_borrow();
            }
            let task: Box<Task> = Local::take();
            cleanup_task(task);
        }).join();
    }

    #[test]
    fn borrow_with_return() {
        Thread::start(proc() {
            let task = box Task::new();
            Local::put(task);

            {
                let _ = Local::borrow(None::<Task>);
            }

            let task: Box<Task> = Local::take();
            cleanup_task(task);
        }).join();
    }

    #[test]
    fn try_take() {
        Thread::start(proc() {
            let task = box Task::new();
            Local::put(task);

            let t: Box<Task> = Local::try_take().unwrap();
            let u: Option<Box<Task>> = Local::try_take();
            assert!(u.is_none());

            cleanup_task(t);
        }).join();
    }

    fn cleanup_task(mut t: Box<Task>) {
        t.destroyed = true;
    }

}
