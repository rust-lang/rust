// revisions: mir thir
// NOTE(skippy) these tests output many duplicates, so deduplicate or they become brittle to changes
// [mir]compile-flags: -Zdeduplicate-diagnostics=yes
// [thir]compile-flags: -Z thir-unsafeck -Zdeduplicate-diagnostics=yes

#![feature(deprecated_safe)]
#![warn(unused_unsafe)]

use std::ffi::{OsStr, OsString};

#[deprecated_safe(since = "1.61.0", note = "reason")]
fn foo0() {} //~ ERROR item must be marked unsafe

#[deprecated_safe(since = "1.61.0", note = "reason")]
trait PreviouslySafeTrait {} //~ ERROR item must be marked unsafe

trait PreviouslySafeFunctions {
    #[deprecated_safe(since = "1.61.0", note = "reason")]
    fn foo0(); //~ ERROR item must be marked unsafe
}

fn main() {}
