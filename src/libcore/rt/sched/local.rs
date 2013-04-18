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

use ptr::mut_null;
use libc::c_void;
use cast::transmute;

use super::Scheduler;
use tls = super::super::thread_local_storage;
#[cfg(test)] use super::super::uvio::UvEventLoop;

/// Give the Scheduler to thread-local storage
pub fn put(sched: ~Scheduler) {
    unsafe {
        let key = tls_key();
        let void_sched: *mut c_void = transmute::<~Scheduler, *mut c_void>(sched);
        tls::set(key, void_sched);
    }
}

/// Take ownership of the Scheduler from thread-local storage
pub fn take() -> ~Scheduler {
    unsafe {
        let key = tls_key();
        let void_sched: *mut c_void = tls::get(key);
        assert!(void_sched.is_not_null());
        let sched = transmute::<*mut c_void, ~Scheduler>(void_sched);
        tls::set(key, mut_null());
        return sched;
    }
}

/// Borrow a mutable reference to the thread-local Scheduler
/// # Safety Note
/// Because this leaves the Scheduler in thread-local storage it is possible
/// For the Scheduler pointer to be aliased
pub unsafe fn borrow() -> &mut Scheduler {
    unsafe {
        let key = tls_key();
        let mut void_sched: *mut c_void = tls::get(key);
        assert!(void_sched.is_not_null());
        {
            let void_sched_ptr = &mut void_sched;
            let sched: &mut ~Scheduler = {
                transmute::<&mut *mut c_void, &mut ~Scheduler>(void_sched_ptr)
            };
            let sched: &mut Scheduler = &mut **sched;
            return sched;
        }
    }
}

fn tls_key() -> tls::Key {
    unsafe {
        let key: *mut c_void = rust_get_sched_tls_key();
        let key: &mut tls::Key = transmute(key);
        return *key;
    }
}

extern {
    fn rust_get_sched_tls_key() -> *mut c_void;
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
        let _scheduler = borrow();
    }
    let _scheduler = take();
}

