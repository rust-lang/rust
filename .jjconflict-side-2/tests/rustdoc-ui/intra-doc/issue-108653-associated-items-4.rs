// This is ensuring that the UI output for associated items works when it's being documented
// from another item.

#![deny(rustdoc::broken_intra_doc_links)]
#![allow(nonstandard_style)]

pub trait Trait {
    type Trait;
}

/// [`Struct::Trait`]
//~^ ERROR both an associated constant and an associated type
pub struct Struct;

impl Trait for Struct {
    type Trait = Struct;
}

impl Struct {
    pub const Trait: usize = 0;
}
