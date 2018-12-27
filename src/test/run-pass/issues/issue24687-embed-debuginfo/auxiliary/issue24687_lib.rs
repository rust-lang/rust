#![crate_type="lib"]

// This is a file that pulls in a separate file as a submodule, where
// that separate file has many multi-byte characters, to try to
// encourage the compiler to trip on them.

mod issue24687_mbcs_in_comments;

pub use issue24687_mbcs_in_comments::D;

