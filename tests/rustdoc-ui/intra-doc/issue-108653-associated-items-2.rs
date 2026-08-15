// This is ensuring that the UI output for associated items is as expected.

#![deny(rustdoc::broken_intra_doc_links)]

/// [`Trait::IDENT`]
//~^ ERROR both an associated constant and an associated type
pub trait Trait {
    type IDENT;
    const IDENT: usize;
}

/// [`Trait2::IDENT`]
//~^ ERROR both an associated function and an associated type
pub trait Trait2 {
    type IDENT;
    fn IDENT() {}
}
