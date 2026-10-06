//! Tests for specific instances of <https://github.com/rust-lang/rust/issues/161621> that don't
//! pass yet. See `dispatchability-placeholder-satisfies-bounds.rs` for context.
// FIXME(@dianne): this should be gone soon
//@ known-bug: unknown

trait HasAssoc {
    type Assoc;
}

trait Parent<T> {}

trait NotDynCompatible
where
    // We need to be able to normalize this for RustaceansAreAwesome in order to prove
    // `<(RustaceansAreAwesome,) as HasAssoc>::Assoc: Owned` for the projection in the argument to
    // the `Self: Parent<...>` supertrait bound. Currently a projection clause for it is missing
    // from the `ParamEnv`.
    (Self,): HasAssoc<Assoc = str>,
    Self: Parent<<<(Self,) as HasAssoc>::Assoc as ToOwned>::Owned>,
{
    fn f(&self);
}

fn main() {}
