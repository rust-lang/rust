#![deny(rustdoc::broken_intra_doc_links)]
#![allow(nonstandard_style)]

/// [`u32::MAX`]
//~^ ERROR both an associated constant and a trait
pub mod u32 {
    pub trait MAX {}
}
