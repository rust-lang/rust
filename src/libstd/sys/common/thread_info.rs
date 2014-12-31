// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use thread::Thread;
use cell::RefCell;
use string::String;

struct ThreadInfo {
    // This field holds the known bounds of the stack in (lo, hi)
    // form. Not all threads necessarily know their precise bounds,
    // hence this is optional.
    stack_bounds: (uint, uint),
    stack_guard: uint,
    thread: Thread,
}

thread_local! { static THREAD_INFO: RefCell<Option<ThreadInfo>> = RefCell::new(None) }

impl ThreadInfo {
    fn with<R>(f: |&mut ThreadInfo| -> R) -> R {
        if THREAD_INFO.destroyed() {
            panic!("Use of std::thread::Thread::current() is not possible after \
                    the thread's local data has been destroyed");
        }

        THREAD_INFO.with(|c| {
            if c.borrow().is_none() {
                *c.borrow_mut() = Some(ThreadInfo {
                    stack_bounds: (0, 0),
                    stack_guard: 0,
                    thread: NewThread::new(None),
                })
            }
            f(c.borrow_mut().as_mut().unwrap())
        })
    }
}

pub fn current_thread() -> Thread {
    ThreadInfo::with(|info| info.thread.clone())
}

pub fn stack_guard() -> uint {
    ThreadInfo::with(|info| info.stack_guard)
}

pub fn set(stack_bounds: (uint, uint), stack_guard: uint, thread: Thread) {
    THREAD_INFO.with(|c| assert!(c.borrow().is_none()));
    THREAD_INFO.with(move |c| *c.borrow_mut() = Some(ThreadInfo{
        stack_bounds: stack_bounds,
        stack_guard: stack_guard,
        thread: thread,
    }));
}

// a hack to get around privacy restrictions; implemented by `std::thread::Thread`
pub trait NewThread {
    fn new(name: Option<String>) -> Self;
}
