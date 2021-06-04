// Regression test for issue #76202
// Tests that we don't ICE when we have a trait impl on a TAIT.

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

trait Dummy {}
impl Dummy for () {}

type F = impl Dummy;
fn f() -> F {}

trait Test {
    fn test(self);
}

impl Test for F { //~ ERROR cannot implement trait
    fn test(self) {}
}

fn main() {
    let x: F = f();
    x.test();
}
