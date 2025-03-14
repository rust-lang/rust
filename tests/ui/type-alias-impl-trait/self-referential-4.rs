#![feature(type_alias_impl_trait)]

type Bar<'a, 'b> = impl PartialEq<Bar<'b, 'static>> + std::fmt::Debug;

#[define_opaque(Bar)]
fn bar<'a, 'b>(i: &'a i32) -> Bar<'a, 'b> {
    i //~^ ERROR can't compare `&i32` with `Bar<'b, 'static>`
}

type Foo<'a, 'b> = impl PartialEq<Foo<'static, 'b>> + std::fmt::Debug;

#[define_opaque(Foo)]
fn foo<'a, 'b>(i: &'a i32) -> Foo<'a, 'b> {
    i //~^ ERROR can't compare `&i32` with `Foo<'static, 'b>`
}

type Moo<'a, 'b> = impl PartialEq<Moo<'static, 'a>> + std::fmt::Debug;

#[define_opaque(Moo)]
fn moo<'a, 'b>(i: &'a i32) -> Moo<'a, 'b> {
    i //~^ ERROR can't compare `&i32` with `Moo<'static, 'a>`
}

fn main() {
    let meh = 42;
    let muh = 69;
    assert_eq!(bar(&meh), bar(&meh));
}
