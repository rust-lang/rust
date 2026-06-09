// Regression test for <https://github.com/rust-lang/rust/issues/113840>
//
// This test makes sure that multiple marker (method-less) traits can reuse the
// same pointer for upcasting.

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

trait A: M0 + M1 + M2 + T {}

trait B: M0 + M1 + T + M2 {}

trait C: M0 + T + M1 + M2 {}

trait D: T + M0 + M1 + M2 {}

struct S;

impl M0 for S {}
impl M1 for S {}
impl M2 for S {}
impl T for S {}

#[rustc_dump_vtable]
impl A for S {}
//~^ ERROR vtable entries

#[rustc_dump_vtable]
impl B for S {}
//~^ ERROR vtable entries

#[rustc_dump_vtable]
impl C for S {}
//~^ ERROR vtable entries

#[rustc_dump_vtable]
impl D for S {}
//~^ ERROR vtable entries

fn main() {}
