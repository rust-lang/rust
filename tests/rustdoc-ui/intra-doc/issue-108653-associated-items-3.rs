// This is ensuring that the UI output for associated items works when it's being documented
// from another item.

#![deny(rustdoc::broken_intra_doc_links)]
#![allow(nonstandard_style)]

pub trait Trait {
    type Trait;
    const Trait: usize;
}

/// [`Trait`]
//~^ ERROR
/// [`Trait::Trait`]
//~^ ERROR
pub const Trait: usize = 0;
