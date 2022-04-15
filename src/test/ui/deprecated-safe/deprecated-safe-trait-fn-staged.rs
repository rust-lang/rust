// aux-build:deprecated-safe.rs
// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![feature(deprecated_safe)]
#![feature(staged_api)]
#![stable(feature = "deprecated-safe-test", since = "1.61.0")]
#![warn(deprecated_safe_in_future, unused_unsafe)]

extern crate deprecated_safe;

use deprecated_safe::DeprSafeFns;
use std::ffi::OsStr;

struct BadImpls;
impl DeprSafeFns for BadImpls {
    fn depr_safe_fn(&self) {} //~ WARN use of associated function `deprecated_safe::DeprSafeFns::depr_safe_fn` without an `unsafe fn` declaration has been deprecated as it is now an unsafe associated function
    fn depr_safe_fn_generic<K: AsRef<OsStr>, V: AsRef<OsStr>>(&self, key: K, value: V) {} //~ WARN use of associated function `deprecated_safe::DeprSafeFns::depr_safe_fn_generic` without an `unsafe fn` declaration has been deprecated as it is now an unsafe associated function
    fn depr_safe_fn_future(&self) {} //~ WARN use of associated function `deprecated_safe::DeprSafeFns::depr_safe_fn_future` without an `unsafe fn` declaration has been deprecated as it is now an unsafe associated function
    fn depr_safe_fn_2015(&self) {} //~ ERROR method `depr_safe_fn_2015` has an incompatible type for trait
    fn depr_safe_fn_2018(&self) {} //~ WARN use of associated function `deprecated_safe::DeprSafeFns::depr_safe_fn_2018` without an `unsafe fn` declaration has been deprecated as it is now an unsafe associated function
}

struct GoodImpls;
impl DeprSafeFns for GoodImpls {
    unsafe fn depr_safe_fn(&self) {}
    unsafe fn depr_safe_fn_generic<K: AsRef<OsStr>, V: AsRef<OsStr>>(&self, key: K, value: V) {}
    unsafe fn depr_safe_fn_future(&self) {}
    unsafe fn depr_safe_fn_2015(&self) {}
    unsafe fn depr_safe_fn_2018(&self) {}
}

fn main() {}
