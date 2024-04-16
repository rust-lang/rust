//@ known-bug: #116947
#![feature(min_specialization)]

trait MySpecTrait {
    fn f();
}

impl<'a, T: ?Sized> MySpecTrait for T {
    default fn f() {}
}

impl<'a, T: ?Sized> MySpecTrait for &'a T {
    fn f() {}
}

fn main() {}
