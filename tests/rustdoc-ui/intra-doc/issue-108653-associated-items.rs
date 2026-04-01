// This is ensuring that the UI output for associated items is as expected.

#![deny(rustdoc::broken_intra_doc_links)]

pub enum Enum {
    IDENT,
}

/// [`Self::IDENT`]
//~^ ERROR both an associated function and an associated type
pub trait Trait {
    type IDENT;
    fn IDENT();
}

/// [`Self::IDENT`]
//~^ ERROR both an associated function and a variant
impl Trait for Enum {
    type IDENT = usize;
    fn IDENT() {}
}

/// [`Self::IDENT2`]
//~^ ERROR both an associated constant and an associated type
pub trait Trait2 {
    type IDENT2;
    const IDENT2: usize;
}

/// [`Self::IDENT2`]
//~^ ERROR both an associated constant and an associated type
impl Trait2 for Enum {
    type IDENT2 = usize;
    const IDENT2: usize = 0;
}
