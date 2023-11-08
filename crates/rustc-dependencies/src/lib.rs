//! A wrapper around rustc internal crates, which enables switching between compiler provided
//! ones and stable ones published in crates.io

#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]

#[cfg(feature = "in-rust-tree")]
extern crate rustc_lexer;

pub mod lexer {
    #[cfg(not(feature = "in-rust-tree"))]
    pub use ::ra_ap_rustc_lexer::*;

    #[cfg(feature = "in-rust-tree")]
    pub use ::rustc_lexer::*;
}

#[cfg(feature = "in-rust-tree")]
extern crate rustc_parse_format;

pub mod parse_format {
    #[cfg(not(feature = "in-rust-tree"))]
    pub use ::ra_ap_rustc_parse_format::*;

    #[cfg(feature = "in-rust-tree")]
    pub use ::rustc_parse_format::*;
}

#[cfg(feature = "in-rust-tree")]
extern crate rustc_abi;

pub mod abi {
    #[cfg(not(feature = "in-rust-tree"))]
    pub use ::ra_ap_rustc_abi::*;

    #[cfg(feature = "in-rust-tree")]
    pub use ::rustc_abi::*;
}

#[cfg(feature = "in-rust-tree")]
extern crate rustc_index;

pub mod index {
    #[cfg(not(feature = "in-rust-tree"))]
    pub use ::ra_ap_rustc_index::*;

    #[cfg(feature = "in-rust-tree")]
    pub use ::rustc_index::*;
}
