#![feature(type_alias_impl_trait)]

fn id(s: &str) -> &str {
    s
}

type Opaque<'a> = impl Sized + 'a;

#[define_opaque(Opaque)]
fn test(s: &str) -> (impl Fn(&str) -> Opaque<'_>, impl Fn(&str) -> Opaque<'_>) {
    (id, id)
    //~^ ERROR expected generic lifetime parameter, found `'_`
    //~| ERROR expected generic lifetime parameter, found `'_`
}

fn id2<'a, 'b>(s: (&'a str, &'b str)) -> (&'a str, &'b str) {
    s
}

type Opaque2<'a> = impl Sized + 'a;

#[define_opaque(Opaque2)]
fn test2() -> impl for<'a, 'b> Fn((&'a str, &'b str)) -> (Opaque2<'a>, Opaque2<'b>) {
    id2
    //~^ ERROR expected generic lifetime parameter, found `'a`
    //~| ERROR expected generic lifetime parameter, found `'b`
}

type Opaque3<'a> = impl Sized + 'a;

#[define_opaque(Opaque3)]
fn test3(s: &str) -> (impl Fn(&str) -> Opaque3<'_>, Opaque3<'_>) {
    (id, s) //~ ERROR expected generic lifetime parameter, found `'_`
}

type Opaque4<'a> = impl Sized + 'a;
#[define_opaque(Opaque4)]
fn test4(s: &str) -> (Opaque4<'_>, impl Fn(&str) -> Opaque4<'_>) {
    (s, id) //~ ERROR expected generic lifetime parameter, found `'_`
}

type Inner<'a> = impl Sized;
#[define_opaque(Inner)]
fn outer_impl() -> impl for<'a> Fn(&'a ()) -> Inner<'a> {
    |x| x //~ ERROR expected generic lifetime parameter, found `'a`
}

fn main() {}
