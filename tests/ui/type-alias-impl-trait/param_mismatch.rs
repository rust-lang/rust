//! This test checks that when checking for opaque types that
//! only differ in lifetimes, we handle the case of non-generic
//! regions correctly.
#![feature(type_alias_impl_trait)]

fn id(s: &str) -> &str {
    s
}
type Opaque<'a> = impl Sized + 'a;
// The second `Opaque<'_>` has a higher kinded lifetime, not a generic parameter
#[define_opaque(Opaque)]
fn test(s: &str) -> (Opaque<'_>, impl Fn(&str) -> Opaque<'_>) {
    (s, id)
    //~^ ERROR: expected generic lifetime parameter, found `'_`
}

fn main() {}
