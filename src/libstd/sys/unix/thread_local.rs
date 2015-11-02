// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)] // sys isn't exported yet

use libc::c_int;
use core::nonzero::NonZero;
use sys::inner::*;
use sys::common::thread_local as sys;

#[derive(Copy, Clone)]
pub struct OsKeyValue(pthread_key_t);
impl_inner!(OsKeyValue(pthread_key_t));

pub type OsKey = sys::OsKey<OsKeyValue>;
pub type StaticOsKey = sys::StaticOsKey<OsKeyValue>;

#[cfg(all(any(target_os = "macos", target_os = "linux"),
          not(target_arch = "aarch64"),
          not(no_elf_tls)))]
pub use self::tls::Key;

#[cfg(any(not(any(target_os = "macos", target_os = "linux")),
          target_arch = "aarch64",
                    no_elf_tls))]
pub type Key<T> = sys::Key<T, OsKeyValue>;

impl OsKeyValue {
    #[inline]
    unsafe fn create_raw(dtor: Option<unsafe extern fn(*mut u8)>) -> pthread_key_t {
        let mut key = 0;
        assert_eq!(pthread_key_create(&mut key, dtor), 0);
        key
    }
}

impl sys::OsKeyImp for OsKeyValue {
    unsafe fn create(dtor: Option<unsafe extern fn(*mut u8)>) -> Self {
        let key = Self::create_raw(dtor);
        OsKeyValue(match key {
            0 => key,
            key => {
                let key_ = Self::create_raw(dtor);
                OsKeyValue(key).destroy();
                key_
            },
        })
    }

    #[inline]
    unsafe fn get(&self) -> *mut u8 {
        pthread_getspecific(*self.as_inner())
    }

    #[inline]
    unsafe fn set(&self, value: *mut u8) {
        let r = pthread_setspecific(self.into_inner(), value);
        debug_assert_eq!(r, 0);
    }

    #[inline]
    unsafe fn destroy(&self) {
        let r = pthread_key_delete(self.into_inner());
        debug_assert_eq!(r, 0);
    }

    unsafe fn from_usize(value: usize) -> Self {
        OsKeyValue(value as pthread_key_t)
    }

    unsafe fn into_usize(&self) -> NonZero<usize> {
        NonZero::new(self.into_inner() as usize)
    }
}

#[cfg(all(any(target_os = "macos", target_os = "linux"),
          not(target_arch = "aarch64"),
          not(no_elf_tls)))]
#[doc(hidden)]
mod tls {
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

        pub unsafe fn get(&'static self) -> Option<&'static UnsafeCell<Option<T>>> {
            if intrinsics::needs_drop::<T>() && self.dtor_running.get() {
                return None
            }
            self.register_dtor();
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

    // Since what appears to be glibc 2.18 this symbol has been shipped which
    // GCC and clang both use to invoke destructors in thread_local globals, so
    // let's do the same!
    //
    // Note, however, that we run on lots older linuxes, as well as cross
    // compiling from a newer linux to an older linux, so we also have a
    // fallback implementation to use as well.
    //
    // Due to rust-lang/rust#18804, make sure this is not generic!
    #[cfg(target_os = "linux")]
    unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern fn(*mut u8)) {
        use prelude::v1::*;
        use mem;
        use libc;
        use sys::thread_local as os;

        extern {
            #[linkage = "extern_weak"]
            static __dso_handle: *mut u8;
            #[linkage = "extern_weak"]
            static __cxa_thread_atexit_impl: *const libc::c_void;
        }
        if !__cxa_thread_atexit_impl.is_null() {
            type F = unsafe extern fn(dtor: unsafe extern fn(*mut u8),
                                      arg: *mut u8,
                                      dso_handle: *mut u8) -> libc::c_int;
            mem::transmute::<*const libc::c_void, F>(__cxa_thread_atexit_impl)
            (dtor, t, &__dso_handle as *const _ as *mut _);
            return
        }

        // The fallback implementation uses a vanilla OS-based TLS key to track
        // the list of destructors that need to be run for this thread. The key
        // then has its own destructor which runs all the other destructors.
        //
        // The destructor for DTORS is a little special in that it has a `while`
        // loop to continuously drain the list of registered destructors. It
        // *should* be the case that this loop always terminates because we
        // provide the guarantee that a TLS key cannot be set after it is
        // flagged for destruction.
        static DTORS: os::StaticOsKey = os::StaticOsKey::new(Some(run_dtors));
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

    // OSX's analog of the above linux function is this _tlv_atexit function.
    // The disassembly of thread_local globals in C++ (at least produced by
    // clang) will have this show up in the output.
    #[cfg(target_os = "macos")]
    unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern fn(*mut u8)) {
        extern {
            fn _tlv_atexit(dtor: unsafe extern fn(*mut u8),
                           arg: *mut u8);
        }
        _tlv_atexit(dtor, t);
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
}

#[cfg(any(target_os = "macos",
          target_os = "ios"))]
type pthread_key_t = ::libc::c_ulong;

#[cfg(any(target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "bitrig",
          target_os = "netbsd",
          target_os = "openbsd",
          target_os = "nacl"))]
type pthread_key_t = ::libc::c_int;

#[cfg(not(any(target_os = "macos",
              target_os = "ios",
              target_os = "freebsd",
              target_os = "dragonfly",
              target_os = "bitrig",
              target_os = "netbsd",
              target_os = "openbsd",
              target_os = "nacl")))]
type pthread_key_t = ::libc::c_uint;

extern {
    fn pthread_key_create(key: *mut pthread_key_t,
                          dtor: Option<unsafe extern fn(*mut u8)>) -> c_int;
    fn pthread_key_delete(key: pthread_key_t) -> c_int;
    fn pthread_getspecific(key: pthread_key_t) -> *mut u8;
    fn pthread_setspecific(key: pthread_key_t, value: *mut u8) -> c_int;
}
