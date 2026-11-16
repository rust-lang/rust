//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no

#![feature(rustc_attrs)]

#[rustc_on_unimplemented(
    message = "cannot add `{Rhs}` to `{Self}`",
    label = "no implementation for `{Self} + {Rhs}`",
)]
pub trait MyAdd<Rhs = Self> {
    fn add(self, rhs: Rhs) -> Self;
}

fn main() {
    MyAdd::add(42_u8, 42.0);
}
