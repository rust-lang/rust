// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)] // stack_guard isn't used right now on all platforms

use cell::RefCell;
use sys::thread::guard::Guard;
use thread::Thread;

struct ThreadInfo {
    stack_guard: Option<Guard>,
    thread: Thread,
}

thread_local! { static THREAD_INFO: RefCell<Option<ThreadInfo>> = RefCell::new(None) }

impl ThreadInfo {
    fn with<R, F>(f: F) -> Option<R> where F: FnOnce(&mut ThreadInfo) -> R {
        THREAD_INFO.try_with(move |c| {
            if c.borrow().is_none() {
                *c.borrow_mut() = Some(ThreadInfo {
                    stack_guard: None,
                    thread: Thread::new(None),
                })
            }
            f(c.borrow_mut().as_mut().unwrap())
        }).ok()
    }
}

pub fn current_thread() -> Option<Thread> {
    ThreadInfo::with(|info| info.thread.clone())
}

pub fn stack_guard() -> Option<Guard> {
    ThreadInfo::with(|info| info.stack_guard.clone()).and_then(|o| o)
}

pub fn set(stack_guard: Option<Guard>, thread: Thread) {
    THREAD_INFO.with(|c| assert!(c.borrow().is_none()));
    THREAD_INFO.with(move |c| *c.borrow_mut() = Some(ThreadInfo{
        stack_guard,
        thread,
    }));
}

pub fn reset_guard(stack_guard: Option<Guard>) {
    THREAD_INFO.with(move |c| c.borrow_mut().as_mut().unwrap().stack_guard = stack_guard);
}
