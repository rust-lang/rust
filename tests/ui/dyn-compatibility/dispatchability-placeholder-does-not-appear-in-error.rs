//! Tests for <https://github.com/rust-lang/rust/issues/161621> that currently don't pass due to
//! other limitations. We at least shouldn't see `RustaceansAreAwesome` mentioned in error messages.
//! If we do see that, and especially if that's all that prevents these from compiling, dyn-
//! compatibility checking (specifically receiver dispatchability checking) needs to be updated.
//!
//! See `dispatchability-placeholder-satisfies-bounds.rs` for additional context.

trait HasAssoc1 {
    type Assoc;
}

trait HasAssoc2 {
    type Assoc;
}

trait Parent<T: ?Sized> {}

// Due to param-env normalization being done in an unnormalized param-env, neither of these examples
// can prove `<Self as HasAssoc1>::Assoc: HasAssoc2`.

trait AliasNormalizesToSelf //~ ERROR: the trait bound `Self: HasAssoc2` is not satisfied
where
    Self: HasAssoc1<Assoc = Self>,
    <Self as HasAssoc1>::Assoc: HasAssoc2<Assoc = Self>,
    Self: Parent<<<Self as HasAssoc1>::Assoc as HasAssoc2>::Assoc>,
{
    fn f(&self); //~ ERROR: the trait bound `Self: HasAssoc2` is not satisfied
}

trait AliasNormalizesToUnit //~ ERROR: the trait bound `(): HasAssoc2` is not satisfied
where
    Self: HasAssoc1<Assoc = ()>,
    <Self as HasAssoc1>::Assoc: HasAssoc2<Assoc = Self>,
    Self: Parent<<<Self as HasAssoc1>::Assoc as HasAssoc2>::Assoc>,
{
    fn f(&self); //~ ERROR: the trait bound `(): HasAssoc2` is not satisfied
}

fn main() {}
