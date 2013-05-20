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
use tls = rt::thread_local_storage;
use unstable::finally::Finally;

#[cfg(test)] use rt::uv::uvio::UvEventLoop;

/// Give the Scheduler to thread-local storage
pub fn put(sched: ~Scheduler) {
    unsafe {
        let key = tls_key();
        let void_sched: *mut c_void = cast::transmute(sched);
        tls::set(key, void_sched);
    }
}

/// Take ownership of the Scheduler from thread-local storage
pub fn take() -> ~Scheduler {
    unsafe {
        let key = tls_key();
        let void_sched: *mut c_void = tls::get(key);
        rtassert!(void_sched.is_not_null());
        let sched: ~Scheduler = cast::transmute(void_sched);
        tls::set(key, mut_null());
        return sched;
    }
}

/// Check whether there is a thread-local Scheduler attached to the running thread
pub fn exists() -> bool {
    unsafe {
        match maybe_tls_key() {
            Some(key) => tls::get(key).is_not_null(),
            None => false
        }
    }
}

/// Borrow the thread-local scheduler from thread-local storage.
/// While the scheduler is borrowed it is not available in TLS.
pub fn borrow(f: &fn(&mut Scheduler)) {
    let mut sched = take();

    // XXX: Need a different abstraction from 'finally' here to avoid unsafety
    unsafe {
        let unsafe_sched = cast::transmute_mut_region(&mut *sched);
        let sched = Cell(sched);

        do (|| {
            f(unsafe_sched);
        }).finally {
            put(sched.take());
        }
    }
}

/// Borrow a mutable reference to the thread-local Scheduler
///
/// # Safety Note
///
/// Because this leaves the Scheduler in thread-local storage it is possible
/// For the Scheduler pointer to be aliased
pub unsafe fn unsafe_borrow() -> *mut Scheduler {
    let key = tls_key();
    let mut void_sched: *mut c_void = tls::get(key);
    rtassert!(void_sched.is_not_null());
    {
        let sched: *mut *mut c_void = &mut void_sched;
        let sched: *mut ~Scheduler = sched as *mut ~Scheduler;
        let sched: *mut Scheduler = &mut **sched;
        return sched;
    }
}

pub unsafe fn unsafe_borrow_io() -> *mut IoFactoryObject {
    let sched = unsafe_borrow();
    let io: *mut IoFactoryObject = (*sched).event_loop.io().unwrap();
    return io;
}

fn tls_key() -> tls::Key {
    match maybe_tls_key() {
        Some(key) => key,
        None => abort!("runtime tls key not initialized")
    }
}

fn maybe_tls_key() -> Option<tls::Key> {
    unsafe {
        let key: *mut c_void = rust_get_rt_tls_key();
        let key: &mut tls::Key = cast::transmute(key);
        let key = *key;
        // Check that the key has been initialized.

        // NB: This is a little racy because, while the key is
        // initalized under a mutex and it's assumed to be initalized
        // in the Scheduler ctor by any thread that needs to use it,
        // we are not accessing the key under a mutex.  Threads that
        // are not using the new Scheduler but still *want to check*
        // whether they are running under a new Scheduler may see a 0
        // value here that is in the process of being initialized in
        // another thread. I think this is fine since the only action
        // they could take if it was initialized would be to check the
        // thread-local value and see that it's not set.
        if key != -1 {
            return Some(key);
        } else {
            return None;
        }
    }
}

extern {
    #[fast_ffi]
    fn rust_get_rt_tls_key() -> *mut c_void;
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
