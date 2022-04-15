// aux-build:deprecated-safe.rs
// revisions: mir thir
// NOTE(skippy) these tests output many duplicates, so deduplicate or they become brittle to changes
// [mir]compile-flags: -Zdeduplicate-diagnostics=yes
// [thir]compile-flags: -Z thir-unsafeck -Zdeduplicate-diagnostics=yes

#![feature(deprecated_safe)]
#![feature(type_alias_impl_trait)]
#![warn(unused_unsafe)]

extern crate deprecated_safe;

use deprecated_safe::DeprSafeFns;

#[deprecated_safe(since = "1.61.0", unsafe_edition = "2015", note = "reason")] //~ ERROR 'unsafe_edition' is invalid outside of rustc
unsafe fn foo0() {}

#[deprecated_safe(since = "1.61.0", unsafe_edition = "BAD", note = "reason")] //~ ERROR 'unsafe_edition' is invalid outside of rustc
unsafe fn foo1() {}

#[deprecated_safe(since = "TBD", unsafe_edition = "BAD", note = "reason")] //~ ERROR 'unsafe_edition' is invalid outside of rustc
unsafe fn foo2() {}

#[deprecated_safe]
unsafe fn foo3() {}

#[deprecated_safe = "hi"]
unsafe fn foo4() {}

#[deprecated_safe(since = "1.61.0")]
unsafe fn foo5() {}

#[deprecated_safe(note = "reason")]
unsafe fn foo6() {}

#[deprecated_safe(since = "1.61.0")]
#[deprecated_safe(note = "reason")] //~ ERROR multiple `deprecated_safe` attributes
unsafe fn foo7() {}

#[deprecated_safe(since = "1.61.0", extra = "")] //~ ERROR unknown meta item 'extra'
unsafe fn foo8() {}

#[deprecated_safe(since = "1.61.0", since = "1.61.0")] //~ ERROR multiple 'since' items
unsafe fn foo9() {}

#[deprecated_safe(since)] //~ ERROR incorrect meta item
unsafe fn foo10() {}

#[deprecated_safe(since = 1_61_0)] //~ ERROR literal in `deprecated_safe` value must be a string
unsafe fn foo11() {}

#[deprecated_safe(1_61_0)] //~ ERROR item in `deprecated_safe` must be a key/value pair
unsafe fn foo12() {}

#[deprecated_safe(since = "1.61.0")] //~ ERROR this `#[deprecated_safe]` annotation has no effect
static FOO0: u32 = 0;

struct BadAnnotation;
impl DeprSafeFns for BadAnnotation {
    #[deprecated_safe(since = "1.61.0", note = "reason")] //~ ERROR this `#[deprecated_safe]` annotation has no effect
    unsafe fn depr_safe_fn(&self) {}
}

fn main() {}
