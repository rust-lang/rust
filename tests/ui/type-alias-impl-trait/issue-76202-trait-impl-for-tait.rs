// Regression test for issue #76202
// Tests that we don't ICE when we have a trait impl on a TAIT.

//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ check-pass
#![feature(type_alias_impl_trait)]

trait Test {
    fn test(self);
}


impl Test for define::F {
    fn test(self) {}
}

// Ok because `i32` does not implement `Dummy`,
// so it can't possibly be the hidden type of `F`.
impl Test for i32 {
    fn test(self) {}
}

mod define {
    use super::*;

    pub trait Dummy {}
    impl Dummy for () {}

    pub type F = impl Dummy;
    pub fn f() -> F {}
}

fn main() {
    let x = define::f();
    x.test();
}
