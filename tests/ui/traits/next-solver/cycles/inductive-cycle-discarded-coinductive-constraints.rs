//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(rustc_attrs, marker_trait_attr)]
#[rustc_coinductive]
trait Trait {}

impl<T, U> Trait for (T, U)
where
    (U, T): Trait,
    (T, U): Inductive,
    (): ConstrainToU32<T>,
{}

trait ConstrainToU32<T> {}
impl ConstrainToU32<u32> for () {}

// We only prefer the candidate without an inductive cycle
// once the inductive cycle has the same constraints as the
// other goal.
#[marker]
trait Inductive {}
impl<T, U> Inductive for (T, U)
where
    (T, U): Trait,
{}

impl Inductive for (u32, u32) {}

fn impls_trait<T, U>()
where
    (T, U): Trait,
{}

fn main() {
    impls_trait::<_, _>();
}
