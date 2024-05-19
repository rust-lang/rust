#![deny(rustdoc::broken_intra_doc_links)]
#![allow(nonstandard_style)]

/// [`u32::MAX`]
//~^ ERROR both an associated constant and a primitive type
pub mod u32 {
    pub use std::primitive::u32 as MAX;
}
