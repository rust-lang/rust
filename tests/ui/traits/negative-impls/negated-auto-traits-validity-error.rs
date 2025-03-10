#![feature(auto_traits, negative_impls)]

auto trait Foo {}

struct AdditionalLt<'a, T>(&'a (), T);
impl<'a, T: 'a> !Foo for AdditionalLt<'a, T> {}
//~^ ERROR `!Foo` impl requires `T: 'a` but the struct it is implemented for does not

struct AdditionalBound<T>(T);
trait Bound {}
impl<T: Bound> !Foo for AdditionalBound<T> {}
//~^ ERROR `!Foo` impl requires `T: Bound` but the struct it is implemented for does not

struct TwoParam<T, U>(T, U);
impl<T> !Foo for TwoParam<T, T> {}
//~^ ERROR `!Foo` impls cannot be specialized

struct ConcreteParam<T>(T);
impl !Foo for ConcreteParam<i32> {}
//~^ ERROR `!Foo` impls cannot be specialized

fn main() {}
