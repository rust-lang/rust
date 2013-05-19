// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Access to a single thread-local pointer

use libc::c_void;
use cast;
use option::{Option, Some, None};
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

pub fn tls_key() -> tls::Key {
    match maybe_tls_key() {
        Some(key) => key,
        None => abort!("runtime tls key not initialized")
    }
}

pub fn maybe_tls_key() -> Option<tls::Key> {
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
