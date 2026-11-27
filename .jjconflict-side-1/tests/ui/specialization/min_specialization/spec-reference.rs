// Check that lifetime parameters are allowed in specializing impls.

//@ check-pass

#![feature(min_specialization)]

trait MySpecTrait {
    fn f();
}

impl<T> MySpecTrait for T {
    default fn f() {}
}

impl<'a, T: ?Sized> MySpecTrait for &'a T {
    fn f() {}
}

fn main() {}
