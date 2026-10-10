//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no

#![feature(rustc_attrs)]

#[rustc_on_unimplemented(
    on(all(Self = "{integer}", Rhs = "{float}"), message = "cannot add a float to an integer",),
    on(all(Self = "{float}", Rhs = "{integer}"), message = "cannot add an integer to a float",),
    message = "cannot add `{Rhs}` to `{Self}`",
    label = "no implementation for `{Self} + {Rhs}`"
)]
pub trait MyAdd<Rhs = Self> {
    fn add(self, rhs: Rhs) -> Self;
}

fn main() {
    MyAdd::add(42_u8, 42.0);
}
