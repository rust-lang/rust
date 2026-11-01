//! Auxiliary file for <https://github.com/rust-lang/rust/issues/24687>.
//! This is a file that pulls in a separate file as a submodule, where
//! that separate file has many multi-byte characters, to try to
//! encourage the compiler to trip on them.
#![crate_type="lib"]

#[path = "cross-crate-multibyte-debuginfo-comments.rs"]
mod issue_24687_mbcs_in_comments;

pub use issue_24687_mbcs_in_comments::D;
