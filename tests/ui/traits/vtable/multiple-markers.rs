// Regression test for <https://github.com/rust-lang/rust/issues/113840>
//
// This test makes sure that multiple marker (method-less) traits can reuse the
// same pointer for upcasting.
//
// build-fail
#![crate_type = "lib"]
#![feature(rustc_attrs)]

// Markers
trait M0 {}
trait M1 {}
trait M2 {}

// Just a trait with a method
trait T {
    fn method(&self) {}
}

#[rustc_dump_vtable]
trait A: M0 + M1 + M2 + T {} //~ error: vtable entries for `<S as A>`:

#[rustc_dump_vtable]
trait B: M0 + M1 + T + M2 {} //~ error: vtable entries for `<S as B>`:

#[rustc_dump_vtable]
trait C: M0 + T + M1 + M2 {} //~ error: vtable entries for `<S as C>`:

#[rustc_dump_vtable]
trait D: T + M0 + M1 + M2 {} //~ error: vtable entries for `<S as D>`:

struct S;

impl M0 for S {}
impl M1 for S {}
impl M2 for S {}
impl T for S {}
impl A for S {}
impl B for S {}
impl C for S {}
impl D for S {}

pub fn require_vtables() {
    fn require_vtables(_: &dyn A, _: &dyn B, _: &dyn C, _: &dyn D) {}

    require_vtables(&S, &S, &S, &S)
}
