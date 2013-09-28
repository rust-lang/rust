// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Access to a single thread-local pointer.
//!
//! The runtime will use this for storing ~Task.
//!
//! XXX: Add runtime checks for usage of inconsistent pointer types.
//! and for overwriting an existing pointer.

use libc::c_void;
use cast;
use ptr;
use cell::Cell;
use option::{Option, Some, None};
use unstable::finally::Finally;
use tls = rt::thread_local_storage;

static mut RT_TLS_KEY: tls::Key = -1;

/// Initialize the TLS key. Other ops will fail if this isn't executed first.
#[fixed_stack_segment]
#[inline(never)]
pub fn init_tls_key() {
    unsafe {
        rust_initialize_rt_tls_key(&mut RT_TLS_KEY);
        extern {
            fn rust_initialize_rt_tls_key(key: *mut tls::Key);
        }
    }
}

/// Give a pointer to thread-local storage.
///
/// # Safety note
///
/// Does not validate the pointer type.
#[inline]
pub unsafe fn put<T>(sched: ~T) {
    let key = tls_key();
    let void_ptr: *mut c_void = cast::transmute(sched);
    tls::set(key, void_ptr);
}

/// Take ownership of a pointer from thread-local storage.
///
/// # Safety note
///
/// Does not validate the pointer type.
#[inline]
pub unsafe fn take<T>() -> ~T {
    let key = tls_key();
    let void_ptr: *mut c_void = tls::get(key);
    if void_ptr.is_null() {
        rtabort!("thread-local pointer is null. bogus!");
    }
    let ptr: ~T = cast::transmute(void_ptr);
    tls::set(key, ptr::mut_null());
    return ptr;
}

/// Take ownership of a pointer from thread-local storage.
///
/// # Safety note
///
/// Does not validate the pointer type.
/// Leaves the old pointer in TLS for speed.
#[inline]
pub unsafe fn unsafe_take<T>() -> ~T {
    let key = tls_key();
    let void_ptr: *mut c_void = tls::get(key);
    if void_ptr.is_null() {
        rtabort!("thread-local pointer is null. bogus!");
    }
    let ptr: ~T = cast::transmute(void_ptr);
    return ptr;
}

/// Check whether there is a thread-local pointer installed.
pub fn exists() -> bool {
    unsafe {
        match maybe_tls_key() {
            Some(key) => tls::get(key).is_not_null(),
            None => false
        }
    }
}

/// Borrow the thread-local value from thread-local storage.
/// While the value is borrowed it is not available in TLS.
///
/// # Safety note
///
/// Does not validate the pointer type.
pub unsafe fn borrow<T>(f: &fn(&mut T)) {
    let mut value = take();

    // XXX: Need a different abstraction from 'finally' here to avoid unsafety
    let unsafe_ptr = cast::transmute_mut_region(&mut *value);
    let value_cell = Cell::new(value);

    do (|| {
        f(unsafe_ptr);
    }).finally {
        put(value_cell.take());
    }
}

/// Borrow a mutable reference to the thread-local value
///
/// # Safety Note
///
/// Because this leaves the value in thread-local storage it is possible
/// For the Scheduler pointer to be aliased
pub unsafe fn unsafe_borrow<T>() -> *mut T {
    let key = tls_key();
    let void_ptr = tls::get(key);
    if void_ptr.is_null() {
        rtabort!("thread-local pointer is null. bogus!");
    }
    void_ptr as *mut T
}

pub unsafe fn try_unsafe_borrow<T>() -> Option<*mut T> {
    match maybe_tls_key() {
        Some(key) => {
            let void_ptr = tls::get(key);
            if void_ptr.is_null() {
                None
            } else {
                Some(void_ptr as *mut T)
            }
        }
        None => None
    }
}

#[inline]
fn tls_key() -> tls::Key {
    match maybe_tls_key() {
        Some(key) => key,
        None => rtabort!("runtime tls key not initialized")
    }
}

#[inline]
#[cfg(not(test))]
pub fn maybe_tls_key() -> Option<tls::Key> {
    unsafe {
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
        if RT_TLS_KEY != -1 {
            return Some(RT_TLS_KEY);
        } else {
            return None;
        }
    }
}

// XXX: The boundary between the running runtime and the testing runtime
// seems to be fuzzy at the moment, and trying to use two different keys
// results in disaster. This should not be necessary.
#[inline]
#[cfg(test)]
pub fn maybe_tls_key() -> Option<tls::Key> {
    unsafe { ::cast::transmute(::realstd::rt::local_ptr::maybe_tls_key()) }
}
