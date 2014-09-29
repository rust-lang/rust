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
//! The runtime will use this for storing Box<Task>.
//!
//! FIXME: Add runtime checks for usage of inconsistent pointer types.
//! and for overwriting an existing pointer.

#![allow(dead_code)]

use core::prelude::*;

use core::mem;
use alloc::boxed::Box;

#[cfg(any(windows, // mingw-w32 doesn't like thread_local things
          target_os = "android", // see #10686
          target_os = "ios"))]
pub use self::native::{init, cleanup, put, take, try_take, unsafe_take, exists,
                       unsafe_borrow, try_unsafe_borrow};

#[cfg(not(any(windows, target_os = "android", target_os = "ios")))]
pub use self::compiled::{init, cleanup, put, take, try_take, unsafe_take, exists,
                         unsafe_borrow, try_unsafe_borrow};

/// Encapsulates a borrowed value. When this value goes out of scope, the
/// pointer is returned.
pub struct Borrowed<T> {
    val: *const (),
}

#[unsafe_destructor]
impl<T> Drop for Borrowed<T> {
    fn drop(&mut self) {
        unsafe {
            if self.val.is_null() {
                rtabort!("Aiee, returning null borrowed object!");
            }
            let val: Box<T> = mem::transmute(self.val);
            put::<T>(val);
            rtassert!(exists());
        }
    }
}

impl<T> Deref<T> for Borrowed<T> {
    fn deref<'a>(&'a self) -> &'a T {
        unsafe { &*(self.val as *const T) }
    }
}

impl<T> DerefMut<T> for Borrowed<T> {
    fn deref_mut<'a>(&'a mut self) -> &'a mut T {
        unsafe { &mut *(self.val as *mut T) }
    }
}

/// Borrow the thread-local value from thread-local storage.
/// While the value is borrowed it is not available in TLS.
///
/// # Safety note
///
/// Does not validate the pointer type.
#[inline]
pub unsafe fn borrow<T>() -> Borrowed<T> {
    let val: *const () = mem::transmute(take::<T>());
    Borrowed {
        val: val,
    }
}

/// Compiled implementation of accessing the runtime local pointer. This is
/// implemented using LLVM's thread_local attribute which isn't necessarily
/// working on all platforms. This implementation is faster, however, so we use
/// it wherever possible.
#[cfg(not(any(windows, target_os = "android", target_os = "ios")))]
pub mod compiled {
    use core::prelude::*;

    use alloc::boxed::Box;
    use core::mem;

    #[cfg(test)]
    pub use realrustrt::shouldnt_be_public::RT_TLS_PTR;

    #[cfg(not(test))]
    #[thread_local]
    pub static mut RT_TLS_PTR: *mut u8 = 0 as *mut u8;

    pub fn init() {}

    pub unsafe fn cleanup() {}

    // Rationale for all of these functions being inline(never)
    //
    // The #[thread_local] annotation gets propagated all the way through to
    // LLVM, meaning the global is specially treated by LLVM to lower it to an
    // efficient sequence of instructions. This also involves dealing with fun
    // stuff in object files and whatnot. Regardless, it turns out this causes
    // trouble with green threads and lots of optimizations turned on. The
    // following case study was done on Linux x86_64, but I would imagine that
    // other platforms are similar.
    //
    // On Linux, the instruction sequence for loading the tls pointer global
    // looks like:
    //
    //      mov %fs:0x0, %rax
    //      mov -0x8(%rax), %rbx
    //
    // This code leads me to believe that (%fs:0x0) is a table, and then the
    // table contains the TLS values for the process. Hence, the slot at offset
    // -0x8 is the task TLS pointer. This leads us to the conclusion that this
    // table is the actual thread local part of each thread. The kernel sets up
    // the fs segment selector to point at the right region of memory for each
    // thread.
    //
    // Optimizations lead me to believe that this code is lowered to these
    // instructions in the LLVM codegen passes, because you'll see code like
    // this when everything is optimized:
    //
    //      mov %fs:0x0, %r14
    //      mov -0x8(%r14), %rbx
    //      // do something with %rbx, the rust Task pointer
    //
    //      ... // <- do more things
    //
    //      mov -0x8(%r14), %rbx
    //      // do something else with %rbx
    //
    // Note that the optimization done here is that the first load is not
    // duplicated during the lower instructions. This means that the %fs:0x0
    // memory location is only dereferenced once.
    //
    // Normally, this is actually a good thing! With green threads, however,
    // it's very possible for the code labeled "do more things" to context
    // switch to another thread. If this happens, then we *must* re-load %fs:0x0
    // because it's changed (we're on a different thread). If we don't re-load
    // the table location, then we'll be reading the original thread's TLS
    // values, not our thread's TLS values.
    //
    // Hence, we never inline these functions. By never inlining, we're
    // guaranteed that loading the table is a local decision which is forced to
    // *always* happen.

    /// Give a pointer to thread-local storage.
    ///
    /// # Safety note
    ///
    /// Does not validate the pointer type.
    #[inline(never)] // see comments above
    pub unsafe fn put<T>(sched: Box<T>) {
        RT_TLS_PTR = mem::transmute(sched)
    }

    /// Take ownership of a pointer from thread-local storage.
    ///
    /// # Safety note
    ///
    /// Does not validate the pointer type.
    #[inline(never)] // see comments above
    pub unsafe fn take<T>() -> Box<T> {
        let ptr = RT_TLS_PTR;
        rtassert!(!ptr.is_null());
        let ptr: Box<T> = mem::transmute(ptr);
        // can't use `as`, due to type not matching with `cfg(test)`
        RT_TLS_PTR = mem::transmute(0u);
        ptr
    }

    /// Optionally take ownership of a pointer from thread-local storage.
    ///
    /// # Safety note
    ///
    /// Does not validate the pointer type.
    #[inline(never)] // see comments above
    pub unsafe fn try_take<T>() -> Option<Box<T>> {
        let ptr = RT_TLS_PTR;
        if ptr.is_null() {
            None
        } else {
            let ptr: Box<T> = mem::transmute(ptr);
            // can't use `as`, due to type not matching with `cfg(test)`
            RT_TLS_PTR = mem::transmute(0u);
            Some(ptr)
        }
    }

    /// Take ownership of a pointer from thread-local storage.
    ///
    /// # Safety note
    ///
    /// Does not validate the pointer type.
    /// Leaves the old pointer in TLS for speed.
    #[inline(never)] // see comments above
    pub unsafe fn unsafe_take<T>() -> Box<T> {
        mem::transmute(RT_TLS_PTR)
    }

    /// Check whether there is a thread-local pointer installed.
    #[inline(never)] // see comments above
    pub fn exists() -> bool {
        unsafe {
            RT_TLS_PTR.is_not_null()
        }
    }

    #[inline(never)] // see comments above
    pub unsafe fn unsafe_borrow<T>() -> *mut T {
        if RT_TLS_PTR.is_null() {
            rtabort!("thread-local pointer is null. bogus!");
        }
        RT_TLS_PTR as *mut T
    }

    #[inline(never)] // see comments above
    pub unsafe fn try_unsafe_borrow<T>() -> Option<*mut T> {
        if RT_TLS_PTR.is_null() {
            None
        } else {
            Some(RT_TLS_PTR as *mut T)
        }
    }
}

/// Native implementation of having the runtime thread-local pointer. This
/// implementation uses the `thread_local_storage` module to provide a
/// thread-local value.
pub mod native {
    use core::prelude::*;

    use alloc::boxed::Box;
    use core::mem;
    use core::ptr;
    use thread_local_storage as tls;

    static mut RT_TLS_KEY: tls::Key = -1;

    /// Initialize the TLS key. Other ops will fail if this isn't executed
    /// first.
    pub fn init() {
        unsafe {
            tls::create(&mut RT_TLS_KEY);
        }
    }

    pub unsafe fn cleanup() {
        rtassert!(RT_TLS_KEY != -1);
        tls::destroy(RT_TLS_KEY);
    }

    /// Give a pointer to thread-local storage.
    ///
    /// # Safety note
    ///
    /// Does not validate the pointer type.
    #[inline]
    pub unsafe fn put<T>(sched: Box<T>) {
        let key = tls_key();
        let void_ptr: *mut u8 = mem::transmute(sched);
        tls::set(key, void_ptr);
    }

    /// Take ownership of a pointer from thread-local storage.
    ///
    /// # Safety note
    ///
    /// Does not validate the pointer type.
    #[inline]
    pub unsafe fn take<T>() -> Box<T> {
        let key = tls_key();
        let void_ptr: *mut u8 = tls::get(key);
        if void_ptr.is_null() {
            rtabort!("thread-local pointer is null. bogus!");
        }
        let ptr: Box<T> = mem::transmute(void_ptr);
        tls::set(key, ptr::null_mut());
        return ptr;
    }

    /// Optionally take ownership of a pointer from thread-local storage.
    ///
    /// # Safety note
    ///
    /// Does not validate the pointer type.
    #[inline]
    pub unsafe fn try_take<T>() -> Option<Box<T>> {
        match maybe_tls_key() {
            Some(key) => {
                let void_ptr: *mut u8 = tls::get(key);
                if void_ptr.is_null() {
                    None
                } else {
                    let ptr: Box<T> = mem::transmute(void_ptr);
                    tls::set(key, ptr::null_mut());
                    Some(ptr)
                }
            }
            None => None
        }
    }

    /// Take ownership of a pointer from thread-local storage.
    ///
    /// # Safety note
    ///
    /// Does not validate the pointer type.
    /// Leaves the old pointer in TLS for speed.
    #[inline]
    pub unsafe fn unsafe_take<T>() -> Box<T> {
        let key = tls_key();
        let void_ptr: *mut u8 = tls::get(key);
        if void_ptr.is_null() {
            rtabort!("thread-local pointer is null. bogus!");
        }
        let ptr: Box<T> = mem::transmute(void_ptr);
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
            // initialized under a mutex and it's assumed to be initialized
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

    #[inline] #[cfg(test)]
    pub fn maybe_tls_key() -> Option<tls::Key> {
        use realrustrt;
        unsafe {
            mem::transmute(realrustrt::shouldnt_be_public::maybe_tls_key())
        }
    }
}
