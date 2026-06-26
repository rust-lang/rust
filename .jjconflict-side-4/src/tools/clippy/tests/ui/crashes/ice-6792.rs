//@ check-pass
//! This is a reproducer for the ICE 6792: https://github.com/rust-lang/rust-clippy/issues/6792.
//! The ICE is caused by using `TyCtxt::type_of(assoc_type_id)`.

trait Trait {
    type Ty;

    fn broken() -> Self::Ty;
}

struct Foo;

impl Trait for Foo {
    type Ty = Foo;

    fn broken() -> Self::Ty {
        Self::Ty {}
    }
}

fn main() {}
