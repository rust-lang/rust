#![feature(type_alias_impl_trait)]

type Foo = impl Fn() -> Foo;

#[define_opaque(Foo)]
fn crash(x: Foo) -> Foo {
    //~^ ERROR overflow evaluating the requirement `<Foo as FnOnce<()>>::Output == Foo`
    x
}

fn main() {}
