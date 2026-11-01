//! Regression test for #122904.

#![feature(type_alias_impl_trait)]

trait T {}

type Alias<'a> = impl T;

struct S;
impl<'a> T for &'a S {}

#[define_opaque(Alias)]
fn with_positive(fun: impl Fn(Alias<'_>)) {
//~^ WARN: function cannot return without recursing
    with_positive(|&n| ());
    //~^ ERROR: cannot move out of a shared reference
}

#[define_opaque(Alias)]
fn main(_: Alias<'_>) {
//~^ ERROR: `main` function has wrong type [E0580]
    with_positive(|&a| ());
    //~^ ERROR: cannot move out of a shared reference
}
