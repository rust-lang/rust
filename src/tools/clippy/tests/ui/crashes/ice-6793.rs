//@ check-pass
//! This is a reproducer for the ICE 6793: https://github.com/rust-lang/rust-clippy/issues/6793.
//! The ICE is caused by using `TyCtxt::type_of(assoc_type_id)`, which is the same as the ICE 6792.

trait Trait {
    type Ty: 'static + Clone;

    fn broken() -> Self::Ty;
}

#[derive(Clone)]
struct MyType {
    x: i32,
}

impl Trait for MyType {
    type Ty = MyType;

    fn broken() -> Self::Ty {
        Self::Ty { x: 1 }
    }
}

fn main() {}
