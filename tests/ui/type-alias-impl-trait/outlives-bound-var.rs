// Here we process outlive obligations involving
// opaque types with bound vars in substs.
// This was an ICE.
//
//@ check-pass
#![feature(type_alias_impl_trait)]

type Ty<'a> = impl Sized + 'a;
fn define<'a>() -> Ty<'a> {}

// Ty<'^0>: 'static
fn test1(_: &'static fn(Ty<'_>)) {}

fn test2() {
    None::<&fn(Ty<'_>)>;
}

fn main() { }
