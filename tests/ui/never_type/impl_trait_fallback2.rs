#![feature(type_alias_impl_trait)]

fn main() {}

trait T {}
impl T for i32 {}

fn should_ret_unit() -> impl T {
    //~^ ERROR `(): T` is not satisfied
    panic!()
}

type Foo = impl T;

#[defines(Foo)]
fn a() -> Foo {
    //~^ ERROR `(): T` is not satisfied
    panic!()
}

#[defines(Foo)]
fn b() -> Foo {
    42
}
