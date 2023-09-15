//! A wrapper around rustc internal crates, which enables switching between compiler provided
//! ones and stable ones published in crates.io

#![cfg_attr(feature = "in-rust-tree", feature(rustc_private))]

#[cfg(feature = "in-rust-tree")]
extern crate rustc_lexer;

#[cfg(feature = "in-rust-tree")]
pub mod lexer {
    pub use ::rustc_lexer::*;
}

#[cfg(not(feature = "in-rust-tree"))]
pub mod lexer {
    pub use ::ra_ap_rustc_lexer::*;
}

#[cfg(feature = "in-rust-tree")]
extern crate rustc_parse_format;

#[cfg(feature = "in-rust-tree")]
pub mod parse_format {
    pub use ::rustc_parse_format::*;
}

#[cfg(not(feature = "in-rust-tree"))]
pub mod parse_format {
    pub use ::ra_ap_rustc_parse_format::*;
}

// Upstream broke this for us so we can't update it
pub mod abi {
    pub use ::hkalbasi_rustc_ap_rustc_abi::*;
}

pub mod index {
    pub use ::hkalbasi_rustc_ap_rustc_index::*;
}
