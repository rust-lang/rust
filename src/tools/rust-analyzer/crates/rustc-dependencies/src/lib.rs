//! A wrapper around rustc internal crates, which enables switching between compiler provided
//! ones and stable ones published in crates.io

pub mod lexer {
    #[cfg(not(feature = "in-rust-tree"))]
    pub use ::ra_ap_rustc_lexer::*;

    #[cfg(feature = "in-rust-tree")]
    pub use ::in_tree_rustc_lexer::*;
}

pub mod parse_format {
    #[cfg(not(feature = "in-rust-tree"))]
    pub use ::ra_ap_rustc_parse_format::*;

    #[cfg(feature = "in-rust-tree")]
    pub use ::in_tree_rustc_parse_format::*;
}

pub mod abi {
    #[cfg(not(feature = "in-rust-tree"))]
    pub use ::ra_ap_rustc_abi::*;

    #[cfg(feature = "in-rust-tree")]
    pub use ::in_tree_rustc_abi::*;
}

pub mod index {
    #[cfg(not(feature = "in-rust-tree"))]
    pub use ::ra_ap_rustc_index::*;

    #[cfg(feature = "in-rust-tree")]
    pub use ::in_tree_rustc_index::*;
}
