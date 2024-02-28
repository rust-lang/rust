#![feature(type_alias_impl_trait)]

fn main() {}

trait T {}
impl T for i32 {}

fn should_ret_unit() -> impl T {
    //~^ ERROR trait `T` is not implemented for `()`
    panic!()
}

type Foo = impl T;

fn a() -> Foo {
    //~^ ERROR trait `T` is not implemented for `()`
    panic!()
}

fn b() -> Foo {
    42
}
