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
#[cfg(stage0)]
#[cfg(windows)]
use ptr;
use cell::Cell;
use option::{Option, Some, None};
use unstable::finally::Finally;
#[cfg(stage0)]
#[cfg(windows)]
use unstable::mutex::{Mutex, MUTEX_INIT};
#[cfg(stage0)]
#[cfg(windows)]
use tls = rt::thread_local_storage;

#[cfg(not(stage0), not(windows), test)]
#[thread_local]
pub use realstd::rt::shouldnt_be_public::RT_TLS_PTR;

#[cfg(not(stage0), not(windows), not(test))]
#[thread_local]
pub static mut RT_TLS_PTR: *mut c_void = 0 as *mut c_void;

#[cfg(stage0)]
#[cfg(windows)]
static mut RT_TLS_KEY: tls::Key = -1;
#[cfg(stage0)]
#[cfg(windows)]
static mut tls_lock: Mutex = MUTEX_INIT;
static mut tls_initialized: bool = false;

/// Initialize the TLS key. Other ops will fail if this isn't executed first.
#[inline(never)]
#[cfg(stage0)]
#[cfg(windows)]
pub fn init_tls_key() {
    unsafe {
        tls_lock.lock();
        if !tls_initialized {
            tls::create(&mut RT_TLS_KEY);
            tls_initialized = true;
        }
        tls_lock.unlock();
    }
}

#[cfg(not(stage0), not(windows))]
pub fn init_tls_key() {
    unsafe {
        tls_initialized = true;
    }
}

#[cfg(windows)]
pub unsafe fn cleanup() {
    // No real use to acquiring a lock around these operations. All we're
    // going to do is destroy the lock anyway which races locking itself. This
    // is why the whole function is labeled as 'unsafe'
    assert!(tls_initialized);
    tls::destroy(RT_TLS_KEY);
    tls_lock.destroy();
    tls_initialized = false;
}

#[cfg(not(windows))]
pub unsafe fn cleanup() {
    assert!(tls_initialized);
    tls_initialized = false;
}

/// Give a pointer to thread-local storage.
///
/// # Safety note
///
/// Does not validate the pointer type.
#[inline]
#[cfg(stage0)]
#[cfg(windows)]
pub unsafe fn put<T>(sched: ~T) {
    let key = tls_key();
    let void_ptr: *mut c_void = cast::transmute(sched);
    tls::set(key, void_ptr);
}

/// Give a pointer to thread-local storage.
///
/// # Safety note
///
/// Does not validate the pointer type.
#[inline]
#[cfg(not(stage0), not(windows))]
pub unsafe fn put<T>(sched: ~T) {
    RT_TLS_PTR = cast::transmute(sched)
}

/// Take ownership of a pointer from thread-local storage.
///
/// # Safety note
///
/// Does not validate the pointer type.
#[inline]
#[cfg(stage0)]
#[cfg(windows)]
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
#[inline]
#[cfg(not(stage0), not(windows))]
pub unsafe fn take<T>() -> ~T {
    let ptr: ~T = cast::transmute(RT_TLS_PTR);
    RT_TLS_PTR = cast::transmute(0); // can't use `as`, due to type not matching with `cfg(test)`
    ptr
}

/// Take ownership of a pointer from thread-local storage.
///
/// # Safety note
///
/// Does not validate the pointer type.
/// Leaves the old pointer in TLS for speed.
#[inline]
#[cfg(stage0)]
#[cfg(windows)]
pub unsafe fn unsafe_take<T>() -> ~T {
    let key = tls_key();
    let void_ptr: *mut c_void = tls::get(key);
    if void_ptr.is_null() {
        rtabort!("thread-local pointer is null. bogus!");
    }
    let ptr: ~T = cast::transmute(void_ptr);
    return ptr;
}

/// Take ownership of a pointer from thread-local storage.
///
/// # Safety note
///
/// Does not validate the pointer type.
/// Leaves the old pointer in TLS for speed.
#[inline]
#[cfg(not(stage0), not(windows))]
pub unsafe fn unsafe_take<T>() -> ~T {
    cast::transmute(RT_TLS_PTR)
}

/// Check whether there is a thread-local pointer installed.
#[cfg(stage0)]
#[cfg(windows)]
pub fn exists() -> bool {
    unsafe {
        match maybe_tls_key() {
            Some(key) => tls::get(key).is_not_null(),
            None => false
        }
    }
}

/// Check whether there is a thread-local pointer installed.
#[cfg(not(stage0), not(windows))]
pub fn exists() -> bool {
    unsafe {
        RT_TLS_PTR.is_not_null()
    }
}

/// Borrow the thread-local value from thread-local storage.
/// While the value is borrowed it is not available in TLS.
///
/// # Safety note
///
/// Does not validate the pointer type.
pub unsafe fn borrow<T>(f: |&mut T|) {
    let mut value = take();

    // XXX: Need a different abstraction from 'finally' here to avoid unsafety
    let unsafe_ptr = cast::transmute_mut_region(&mut *value);
    let value_cell = Cell::new(value);

    (|| f(unsafe_ptr)).finally(|| put(value_cell.take()));
}

/// Borrow a mutable reference to the thread-local value
///
/// # Safety Note
///
/// Because this leaves the value in thread-local storage it is possible
/// For the Scheduler pointer to be aliased
#[cfg(stage0)]
#[cfg(windows)]
pub unsafe fn unsafe_borrow<T>() -> *mut T {
    let key = tls_key();
    let void_ptr = tls::get(key);
    if void_ptr.is_null() {
        rtabort!("thread-local pointer is null. bogus!");
    }
    void_ptr as *mut T
}

#[cfg(not(stage0), not(windows))]
pub unsafe fn unsafe_borrow<T>() -> *mut T {
    if RT_TLS_PTR.is_null() {
        rtabort!("thread-local pointer is null. bogus!");
    }
    RT_TLS_PTR as *mut T
}

#[cfg(stage0)]
#[cfg(windows)]
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

#[cfg(not(stage0), not(windows))]
pub unsafe fn try_unsafe_borrow<T>() -> Option<*mut T> {
    if RT_TLS_PTR.is_null() {
        None
    } else {
        Some(RT_TLS_PTR as *mut T)
    }
}

#[inline]
#[cfg(stage0)]
#[cfg(windows)]
fn tls_key() -> tls::Key {
    match maybe_tls_key() {
        Some(key) => key,
        None => rtabort!("runtime tls key not initialized")
    }
}

#[inline]
#[cfg(not(test), stage0)]
#[cfg(not(test), windows)]
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

#[inline]
#[cfg(test, stage0)]
#[cfg(test, windows)]
pub fn maybe_tls_key() -> Option<tls::Key> {
    unsafe { ::cast::transmute(::realstd::rt::shouldnt_be_public::maybe_tls_key()) }
}
