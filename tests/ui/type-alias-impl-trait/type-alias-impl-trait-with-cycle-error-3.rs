#![feature(type_alias_impl_trait)]

type Foo<'a> = impl Fn() -> Foo<'a>;
#[define_opaque(Foo)]
fn crash<'a>(_: &'a (), x: Foo<'a>) -> Foo<'a> {
    //~^ ERROR overflow evaluating the requirement `<Foo<'_> as FnOnce<()>>::Output == Foo<'a>`
    x
}

fn main() {}
