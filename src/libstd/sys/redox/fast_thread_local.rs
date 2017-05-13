// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![cfg(target_thread_local)]
#![unstable(feature = "thread_local_internals", issue = "0")]

use cell::{Cell, UnsafeCell};
use intrinsics;
use ptr;

pub struct Key<T> {
    inner: UnsafeCell<Option<T>>,

    // Metadata to keep track of the state of the destructor. Remember that
    // these variables are thread-local, not global.
    dtor_registered: Cell<bool>,
    dtor_running: Cell<bool>,
}

unsafe impl<T> ::marker::Sync for Key<T> { }

impl<T> Key<T> {
    pub const fn new() -> Key<T> {
        Key {
            inner: UnsafeCell::new(None),
            dtor_registered: Cell::new(false),
            dtor_running: Cell::new(false)
        }
    }

    pub fn get(&'static self) -> Option<&'static UnsafeCell<Option<T>>> {
        unsafe {
            if intrinsics::needs_drop::<T>() && self.dtor_running.get() {
                return None
            }
            self.register_dtor();
        }
        Some(&self.inner)
    }

    unsafe fn register_dtor(&self) {
        if !intrinsics::needs_drop::<T>() || self.dtor_registered.get() {
            return
        }

        register_dtor(self as *const _ as *mut u8,
                      destroy_value::<T>);
        self.dtor_registered.set(true);
    }
}

unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern fn(*mut u8)) {
    // The fallback implementation uses a vanilla OS-based TLS key to track
    // the list of destructors that need to be run for this thread. The key
    // then has its own destructor which runs all the other destructors.
    //
    // The destructor for DTORS is a little special in that it has a `while`
    // loop to continuously drain the list of registered destructors. It
    // *should* be the case that this loop always terminates because we
    // provide the guarantee that a TLS key cannot be set after it is
    // flagged for destruction.
    use sys_common::thread_local as os;

    static DTORS: os::StaticKey = os::StaticKey::new(Some(run_dtors));
    type List = Vec<(*mut u8, unsafe extern fn(*mut u8))>;
    if DTORS.get().is_null() {
        let v: Box<List> = box Vec::new();
        DTORS.set(Box::into_raw(v) as *mut u8);
    }
    let list: &mut List = &mut *(DTORS.get() as *mut List);
    list.push((t, dtor));

    unsafe extern fn run_dtors(mut ptr: *mut u8) {
        while !ptr.is_null() {
            let list: Box<List> = Box::from_raw(ptr as *mut List);
            for &(ptr, dtor) in list.iter() {
                dtor(ptr);
            }
            ptr = DTORS.get();
            DTORS.set(ptr::null_mut());
        }
    }
}

pub unsafe extern fn destroy_value<T>(ptr: *mut u8) {
    let ptr = ptr as *mut Key<T>;
    // Right before we run the user destructor be sure to flag the
    // destructor as running for this thread so calls to `get` will return
    // `None`.
    (*ptr).dtor_running.set(true);

    // The OSX implementation of TLS apparently had an odd aspect to it
    // where the pointer we have may be overwritten while this destructor
    // is running. Specifically if a TLS destructor re-accesses TLS it may
    // trigger a re-initialization of all TLS variables, paving over at
    // least some destroyed ones with initial values.
    //
    // This means that if we drop a TLS value in place on OSX that we could
    // revert the value to its original state halfway through the
    // destructor, which would be bad!
    //
    // Hence, we use `ptr::read` on OSX (to move to a "safe" location)
    // instead of drop_in_place.
    if cfg!(target_os = "macos") {
        ptr::read((*ptr).inner.get());
    } else {
        ptr::drop_in_place((*ptr).inner.get());
    }
}
