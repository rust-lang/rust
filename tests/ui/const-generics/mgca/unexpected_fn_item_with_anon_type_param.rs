// Regression test for https://github.com/rust-lang/rust/issues/158152
//
// Using a function with an anonymous type parameter as a const parameter would ICE because the
// anonymous type parameter would be lowered to a type parameter when it should error if being used
// as a const generic argument, later it would fail an assertion in trait matching because it would
// try to instantiate with no provided arguments when the function has an anonymous type parameter

#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

trait A<T> {}
trait Trait {}

impl A<[usize; fn_item]> for () {}
//~^ ERROR cannot infer anonymous type parameter of `fn_item`

fn fn_item(_: impl Trait) {}

fn main() {}
