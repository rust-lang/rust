#![deny(rustdoc::broken_intra_doc_links)]
#![allow(nonstandard_style)]

/// [`u32::MAX`]
//~^ ERROR both an associated constant and an associated type
pub trait T {
    type MAX;
}

impl T for u32 {
    type MAX = ();
}
