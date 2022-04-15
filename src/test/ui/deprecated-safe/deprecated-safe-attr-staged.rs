// aux-build:deprecated-safe.rs
// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![feature(deprecated_safe)]
#![feature(staged_api)]
#![stable(feature = "deprecated-safe-test", since = "1.61.0")]
#![warn(deprecated_safe_in_future, unused_unsafe)]

extern crate deprecated_safe;

use deprecated_safe::DeprSafeFns;

#[deprecated_safe(since = "1.61.0", unsafe_edition = "2015", note = "reason")]
unsafe fn foo0() {}

#[deprecated_safe(since = "1.61.0", unsafe_edition = "BAD", note = "reason")] //~ ERROR invalid 'unsafe_edition' specified
unsafe fn foo1() {}

#[deprecated_safe(since = "TBD", unsafe_edition = "BAD", note = "reason")] //~ ERROR invalid 'unsafe_edition' specified
unsafe fn foo2() {}

#[deprecated_safe] //~ ERROR missing 'since'
unsafe fn foo3() {}

#[deprecated_safe = "hi"] //~ ERROR missing 'since'
unsafe fn foo4() {}

#[deprecated_safe(since = "1.61.0")] //~ ERROR missing 'note'
unsafe fn foo5() {}

#[deprecated_safe(note = "reason")] //~ ERROR missing 'since'
unsafe fn foo6() {}

struct BadAnnotation;
impl DeprSafeFns for BadAnnotation {
    #[deprecated_safe(since = "1.61.0", note = "reason")] //~ ERROR this `#[deprecated_safe]` annotation has no effect
    unsafe fn depr_safe_fn(&self) {}
}

fn main() {}
