// issue: rust-lang/rust#107228
// ICE broken MIR in DropGlue
//@ compile-flags: -Zvalidate-mir
//@ check-pass

#![feature(specialization)]
#![crate_type="lib"]
#![allow(incomplete_features)]

pub(crate) trait SpecTrait {
    type Assoc;
}

impl<C> SpecTrait for C {
    default type Assoc = Vec<Self>;
}

pub(crate) struct AssocWrap<C: SpecTrait> {
    _assoc: C::Assoc,
}

fn instantiate<C: SpecTrait>() -> AssocWrap<C> {
    loop {}
}

pub fn main() {
    instantiate::<()>();
}
