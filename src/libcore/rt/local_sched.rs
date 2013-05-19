// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Access to the thread-local Scheduler

use prelude::*;
use ptr::mut_null;
use libc::c_void;
use cast;
use cell::Cell;

use rt::sched::Scheduler;
use rt::rtio::{EventLoop, IoFactoryObject};
use unstable::finally::Finally;
use rt::local_ptr;
use tls = rt::thread_local_storage;

#[cfg(test)] use rt::uv::uvio::UvEventLoop;

/// Give the Scheduler to thread-local storage
pub fn put(sched: ~Scheduler) { unsafe { local_ptr::put(sched) } }

/// Take ownership of the Scheduler from thread-local storage
pub fn take() -> ~Scheduler { unsafe { local_ptr::take() } }

/// Check whether there is a thread-local Scheduler attached to the running thread
pub fn exists() -> bool { local_ptr::exists() }

/// Borrow the thread-local scheduler from thread-local storage.
/// While the scheduler is borrowed it is not available in TLS.
pub fn borrow(f: &fn(&mut Scheduler)) { unsafe { local_ptr::borrow(f) } }

/// Borrow a mutable reference to the thread-local Scheduler
///
/// # Safety Note
///
/// Because this leaves the Scheduler in thread-local storage it is possible
/// For the Scheduler pointer to be aliased
pub unsafe fn unsafe_borrow() -> *mut Scheduler { local_ptr::unsafe_borrow() }

pub unsafe fn unsafe_borrow_io() -> *mut IoFactoryObject {
    let sched = unsafe_borrow();
    let io: *mut IoFactoryObject = (*sched).event_loop.io().unwrap();
    return io;
}

#[test]
fn thread_local_scheduler_smoke_test() {
    let scheduler = ~UvEventLoop::new_scheduler();
    put(scheduler);
    let _scheduler = take();
}

#[test]
fn thread_local_scheduler_two_instances() {
    let scheduler = ~UvEventLoop::new_scheduler();
    put(scheduler);
    let _scheduler = take();
    let scheduler = ~UvEventLoop::new_scheduler();
    put(scheduler);
    let _scheduler = take();
}

#[test]
fn borrow_smoke_test() {
    let scheduler = ~UvEventLoop::new_scheduler();
    put(scheduler);
    unsafe {
        let _scheduler = unsafe_borrow();
    }
    let _scheduler = take();
}
