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

/// Initialize the TLS key. Other ops will fail if this isn't executed first.
pub fn init_tls_key() {
    unsafe {
        rust_initialize_rt_tls_key();
        extern {
            fn rust_initialize_rt_tls_key();
        }
    }
}

/// Give a pointer to thread-local storage.
///
/// # Safety note
///
/// Does not validate the pointer type.
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
pub unsafe fn take<T>() -> ~T {
    let key = tls_key();
    let void_ptr: *mut c_void = tls::get(key);
    rtassert!(void_ptr.is_not_null());
    let ptr: ~T = cast::transmute(void_ptr);
    tls::set(key, ptr::mut_null());
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

/// Borrow the thread-local scheduler from thread-local storage.
/// While the scheduler is borrowed it is not available in TLS.
///
/// # Safety note
///
/// Does not validate the pointer type.
pub unsafe fn borrow<T>(f: &fn(&mut T)) {
    let mut value = take();

    // XXX: Need a different abstraction from 'finally' here to avoid unsafety
    let unsafe_ptr = cast::transmute_mut_region(&mut *value);
    let value_cell = Cell(value);

    do (|| {
        f(unsafe_ptr);
    }).finally {
        put(value_cell.take());
    }
}

/// Borrow a mutable reference to the thread-local Scheduler
///
/// # Safety Note
///
/// Because this leaves the Scheduler in thread-local storage it is possible
/// For the Scheduler pointer to be aliased
pub unsafe fn unsafe_borrow<T>() -> *mut T {
    let key = tls_key();
    let mut void_sched: *mut c_void = tls::get(key);
    rtassert!(void_sched.is_not_null());
    {
        let sched: *mut *mut c_void = &mut void_sched;
        let sched: *mut ~T = sched as *mut ~T;
        let sched: *mut T = &mut **sched;
        return sched;
    }
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

    extern {
        #[fast_ffi]
        fn rust_get_rt_tls_key() -> *mut c_void;
    }

}
