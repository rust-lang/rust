#![feature(type_alias_impl_trait)]

type Bar<'a, 'b> = impl PartialEq<Bar<'b, 'a>> + std::fmt::Debug;

#[define_opaque(Bar)]
fn bar<'a, 'b>(i: &'a i32) -> Bar<'a, 'b> {
    //~^ ERROR can't compare `&i32` with `Bar<'b, 'a>`
    i
}

type Foo<'a, 'b> = (i32, impl PartialEq<Foo<'a, 'b>> + std::fmt::Debug);

#[define_opaque(Foo)]
fn foo<'a, 'b>(i: &'a i32) -> Foo<'a, 'b> {
    //~^ ERROR can't compare `&i32` with `(i32, Foo<'a, 'b>::{opaque#0}<'a, 'b>)`
    (42, i)
}

type Moo<'a, 'b> = (i32, impl PartialEq<Moo<'b, 'a>> + std::fmt::Debug);

#[define_opaque(Moo)]
fn moo<'a, 'b>(i: &'a i32) -> Moo<'a, 'b> {
    //~^ ERROR can't compare `&i32` with `(i32, Moo<'b, 'a>::{opaque#0}<'b, 'a>)`
    (42, i)
}

fn main() {
    let meh = 42;
    let muh = 69;
    assert_eq!(bar(&meh), bar(&meh));
}
