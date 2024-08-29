// Here we process outlive obligations involving
// opaque types with bound vars in substs.
// This was an ICE.
//
//@ check-pass
#![feature(type_alias_impl_trait)]

mod tait {
    pub type Ty<'a> = impl Sized + 'a;
    fn define<'a>() -> Ty<'a> {}
}
use tait::Ty;

// Ty<'^0>: 'static
fn test1(_: &'static fn(Ty<'_>)) {}

fn test2() {
    None::<&fn(Ty<'_>)>;
}

fn main() {}
