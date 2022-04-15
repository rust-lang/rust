// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![feature(deprecated_safe)]
#![warn(unused_unsafe)]

use std::ffi::{OsStr, OsString};

#[deprecated_safe(since = "1.61.0", note = "reason")]
fn foo0() {} //~ ERROR item must be marked unsafe

#[deprecated_safe(since = "1.61.0", note = "reason")]
trait PreviouslySafeTrait {} //~ ERROR item must be marked unsafe

#[deprecated_safe(since = "99.99.99", note = "reason")]
fn foo0_future() {} //~ ERROR item must be marked unsafe

#[deprecated_safe(since = "99.99.99", note = "reason")]
trait PreviouslySafeTraitFuture {} //~ ERROR item must be marked unsafe

#[deprecated_safe(since = "TBD", note = "reason")]
fn foo0_tbd() {} //~ ERROR item must be marked unsafe

#[deprecated_safe(since = "TBD", note = "reason")]
trait PreviouslySafeTraitTbd {} //~ ERROR item must be marked unsafe

trait PreviouslySafeFunctions {
    #[deprecated_safe(since = "1.61.0", note = "reason")]
    fn foo0(); //~ ERROR item must be marked unsafe

    #[deprecated_safe(since = "99.99.99", note = "reason")]
    fn foo1(); //~ ERROR item must be marked unsafe

    #[deprecated_safe(since = "TBD", note = "reason")]
    fn foo2(); //~ ERROR item must be marked unsafe
}

fn main() {}
