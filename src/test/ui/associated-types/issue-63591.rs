// check-pass

#![feature(associated_type_bounds)]
// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

fn main() {}

trait Bar { type Assoc; }

trait Thing {
    type Out;
    fn func() -> Self::Out;
}

struct AssocIsCopy;
impl Bar for AssocIsCopy { type Assoc = u8; }

impl Thing for AssocIsCopy {
    type Out = impl Bar<Assoc: Copy>;

    fn func() -> Self::Out {
        AssocIsCopy
    }
}
