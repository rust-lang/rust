// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(missing_docs)]

use boxed::Box;
use sync::Once;
use sys;

macro_rules! rtabort {
    ($($t:tt)*) => (::sys_common::util::abort(format_args!($($t)*)))
}

macro_rules! rtassert {
    ($e:expr) => ({
        if !$e {
            rtabort!(concat!("assertion failed: ", stringify!($e)))
        }
    })
}

pub mod args;
pub mod at_exit_imp;
pub mod backtrace;
pub mod condvar;
pub mod dwarf;
pub mod io;
pub mod libunwind;
pub mod mutex;
pub mod net;
pub mod poison;
pub mod remutex;
pub mod rwlock;
pub mod thread;
pub mod thread_info;
pub mod thread_local;
pub mod unwind;
pub mod util;
pub mod wtf8;

#[cfg(any(all(unix, not(any(target_os = "macos", target_os = "ios", target_os = "emscripten"))),
          all(windows, target_env = "gnu")))]
pub mod gnu;

// common error constructors

/// A trait for viewing representations from std types
#[doc(hidden)]
pub trait AsInner<Inner: ?Sized> {
    fn as_inner(&self) -> &Inner;
}

/// A trait for viewing representations from std types
#[doc(hidden)]
pub trait AsInnerMut<Inner: ?Sized> {
    fn as_inner_mut(&mut self) -> &mut Inner;
}

/// A trait for extracting representations from std types
#[doc(hidden)]
pub trait IntoInner<Inner> {
    fn into_inner(self) -> Inner;
}

/// A trait for creating std types from internal representations
#[doc(hidden)]
pub trait FromInner<Inner> {
    fn from_inner(inner: Inner) -> Self;
}

/// Enqueues a procedure to run when the main thread exits.
///
/// Currently these closures are only run once the main *Rust* thread exits.
/// Once the `at_exit` handlers begin running, more may be enqueued, but not
/// infinitely so. Eventually a handler registration will be forced to fail.
///
/// Returns `Ok` if the handler was successfully registered, meaning that the
/// closure will be run once the main thread exits. Returns `Err` to indicate
/// that the closure could not be registered, meaning that it is not scheduled
/// to be run.
pub fn at_exit<F: FnOnce() + Send + 'static>(f: F) -> Result<(), ()> {
    if at_exit_imp::push(Box::new(f)) {Ok(())} else {Err(())}
}

/// One-time runtime cleanup.
pub fn cleanup() {
    static CLEANUP: Once = Once::new();
    CLEANUP.call_once(|| unsafe {
        args::cleanup();
        sys::stack_overflow::cleanup();
        at_exit_imp::cleanup();
    });
}

// Computes (value*numer)/denom without overflow, as long as both
// (numer*denom) and the overall result fit into i64 (which is the case
// for our time conversions).
#[allow(dead_code)] // not used on all platforms
pub fn mul_div_u64(value: u64, numer: u64, denom: u64) -> u64 {
    let q = value / denom;
    let r = value % denom;
    // Decompose value as (value/denom*denom + value%denom),
    // substitute into (value*numer)/denom and simplify.
    // r < denom, so (denom*numer) is the upper bound of (r*numer)
    q * numer + r * numer / denom
}

#[test]
fn test_muldiv() {
    assert_eq!(mul_div_u64( 1_000_000_000_001, 1_000_000_000, 1_000_000),
               1_000_000_000_001_000);
}
