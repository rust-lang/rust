//! This test checks that when checking for opaque types that
//! only differ in lifetimes, we handle the case of non-generic
//! regions correctly.
#![feature(type_alias_impl_trait)]

fn id2<'a, 'b>(s: (&'a str, &'b str)) -> (&'a str, &'b str) {
    s
}

type Opaque<'a> = impl Sized + 'a;

#[define_opaque(Opaque)]
fn test() -> impl for<'a, 'b> Fn((&'a str, &'b str)) -> (Opaque<'a>, Opaque<'b>) {
    id2
    //~^ ERROR expected generic lifetime parameter, found `'a`
    //~| ERROR expected generic lifetime parameter, found `'b`
}

fn id(s: &str) -> &str {
    s
}

type Opaque2<'a> = impl Sized + 'a;

#[define_opaque(Opaque2)]
fn test2(s: &str) -> (impl Fn(&str) -> Opaque2<'_>, Opaque2<'_>) {
    (id, s) //~ ERROR: expected generic lifetime parameter, found `'_`
}

fn main() {}
