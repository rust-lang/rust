// Check that evaluation of needs_drop<T> fails when T is not monomorphic.
#![feature(generic_const_exprs)]
#![allow(const_evaluatable_unchecked)]
#![allow(incomplete_features)]

struct Bool<const B: bool> {}
impl Bool<true> {
    fn assert() {}
}
fn f<T>() {
    Bool::<{ std::mem::needs_drop::<T>() }>::assert();
    //~^ ERROR no associated function or constant named `assert` found
    //~| ERROR unconstrained generic constant
}
fn main() {
    f::<u32>();
}
