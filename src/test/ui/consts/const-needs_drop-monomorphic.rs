// Check that evaluation of needs_drop<T> fails when T is not monomorphic.
#![feature(const_generics)]
#![allow(const_evaluatable_unchecked)]
#![allow(incomplete_features)]

struct Bool<const B: bool> {}
impl Bool<true> {
    fn assert() {}
}
fn f<T>() {
    Bool::<{ std::mem::needs_drop::<T>() }>::assert();
    //~^ ERROR no function or associated item named `assert` found
    //~| ERROR constant expression depends on a generic parameter
}
fn main() {
    f::<u32>();
}
