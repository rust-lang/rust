// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![feature(deprecated_safe)]
#![feature(staged_api)]
#![stable(feature = "deprecated-safe-test", since = "1.61.0")]
#![warn(deprecated_safe_in_future, unused_unsafe)]

use std::ffi::{OsStr, OsString};

#[deprecated_safe(since = "1.61.0", note = "reason")]
fn foo0() {} //~ ERROR item must be marked unsafe

#[deprecated_safe(since = "1.61.0", note = "reason")]
trait PreviouslySafeTrait {} //~ ERROR item must be marked unsafe

#[deprecated_safe(since = "99.99.99", note = "reason")]
fn foo0_future() {}

#[deprecated_safe(since = "99.99.99", note = "reason")]
trait PreviouslySafeTraitFuture {}

#[deprecated_safe(since = "TBD", note = "reason")]
fn foo0_tbd() {}

#[deprecated_safe(since = "TBD", note = "reason")]
trait PreviouslySafeTraitTbd {}

trait PreviouslySafeFunctions {
    #[deprecated_safe(since = "1.61.0", note = "reason")]
    fn foo0(); //~ ERROR item must be marked unsafe

    #[deprecated_safe(since = "99.99.99", note = "reason")]
    unsafe fn foo1(); //~ ERROR item must not be marked unsafe

    #[deprecated_safe(since = "TBD", note = "reason")]
    unsafe fn foo2(); //~ ERROR item must not be marked unsafe
}

fn main() {}
