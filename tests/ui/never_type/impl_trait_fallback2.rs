#![feature(type_alias_impl_trait)]

fn main() {}

trait T {}
impl T for i32 {}

fn should_ret_unit() -> impl T {
    //~^ ERROR `(): T` is not satisfied
    panic!()
}

type Foo = impl T;

#[define_opaque(Foo)]
fn a() -> Foo {
    //~^ ERROR `(): T` is not satisfied
    panic!()
}

#[define_opaque(Foo)]
fn b() -> Foo {
    42
}
