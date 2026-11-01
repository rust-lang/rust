// Regression test for issue #76202
// Tests that we don't ICE when we have a trait impl on a TAIT.

//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass
#![feature(type_alias_impl_trait)]

pub trait Dummy {}
impl Dummy for () {}
pub type F = impl Dummy;
#[define_opaque(F)]
pub fn f() -> F {}

trait Test {
    fn test(self);
}

impl Test for F {
    fn test(self) {}
}

// Ok because `i32` does not implement `Dummy`,
// so it can't possibly be the hidden type of `F`.
impl Test for i32 {
    fn test(self) {}
}

pub trait Dummy2 {}
impl Dummy2 for () {}

pub type F2 = impl Dummy2;
#[define_opaque(F2)]
pub fn f2() -> F2 {}

fn main() {
    let x = f();
    x.test();
}
