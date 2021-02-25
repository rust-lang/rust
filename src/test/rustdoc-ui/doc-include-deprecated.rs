// aux-build:doc-include-deprecated.rs
// check-pass
#![feature(external_doc)]
#![doc(include = "auxiliary/docs.md")] //~ WARNING deprecated

extern crate inner;

pub use inner::HasDocInclude;

#[doc(include = "auxiliary/docs.md")] //~ WARNING deprecated
pub use inner::HasDocInclude as _;

#[doc(include = "auxiliary/docs.md")] //~ WARNING deprecated
pub use inner::NoDocs;
