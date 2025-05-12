#![feature(type_alias_impl_trait)]

type Opaque<'a> = impl Sized + 'a;

fn foo<'a>(x: &'a ()) -> &'a () {
    x
}

#[define_opaque(Opaque)]
fn test() -> for<'a> fn(&'a ()) -> Opaque<'a> {
    foo //~ ERROR: mismatched types
}

fn main() {}
