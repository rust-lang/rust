// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct ThreadInfo {
    // This field holds the known bounds of the stack in (lo, hi)
    // form. Not all threads necessarily know their precise bounds,
    // hence this is optional.
    stack_bounds: (uint, uint),
    stack_guard: uint,
    unwinder: Unwinder,
    thread: Thread,
}

thread_local!(static THREAD_INFO: RefCell<Option<ThreadInfo>> = RefCell::new(None));

impl ThreadInfo {
    fn with<R>(f: |&ThreadInfo| -> R) -> R {
        THREAD_INFO.with(|c| {
            if c.borrow().is_none() {
                *c.borrow_mut() = Some(ThreadInfo {
                    stack_bounds: (0, 0),
                    stack_guard: 0,
                    unwinder: Unwinder::new(),
                    thread: Thread::new(None),
                })
            }
            f(c.borrow().as_ref().unwrap())
        })
    }
}

pub fn current_thread() -> Thread {
    ThreadInfo::with(|info| info.thread.clone())
}

pub fn panicking() -> bool {
    ThreadInfo::with(|info| info.unwinder.unwinding())
}

pub fn set(stack_bounds: (uint, uint), stack_guard: uint, thread: Thread) {
    THREAD_INFO.with(|c| assert!(c.borrow().is_none()));
    THREAD_INFO.with(|c| *c.borrow_mut() = Some(ThreadInfo{
        stack_bounds: stack_bounds,
        stack_guard: stack_guard,
        unwinder: Unwinder::new(),
        thread: thread,
    }));
}

// a hack to get around privacy restrictions; implemented by `std::thread::Thread`
pub trait NewThread {
    fn new(name: Option<String>) -> Self;
}
