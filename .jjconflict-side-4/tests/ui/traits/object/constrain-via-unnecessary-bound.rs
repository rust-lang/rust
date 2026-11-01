//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/140645>.
// Test that we lower impossible-to-satisfy associated type bounds, which
// may for example constrain impl parameters.

pub trait Other {}

pub trait Trait {
    type Assoc
    where
        Self: Sized;
}

impl Other for dyn Trait {}
// `dyn Trait<Assoc = ()>` is a different "nominal type" than `dyn Trait`.
impl Other for dyn Trait<Assoc = ()> {}
//~^ WARN unnecessary associated type bound for dyn-incompatible associated type

// I hope it's clear that `dyn Trait` (w/o `Assoc`) wouldn't match this impl.
impl<T> dyn Trait<Assoc = T> {}
//~^ WARN unnecessary associated type bound for dyn-incompatible associated type

fn main() {}
