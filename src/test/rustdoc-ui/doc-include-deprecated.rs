// aux-build:doc-include-deprecated.rs
// check-pass
#![feature(external_doc)]
#![doc(include = "auxiliary/docs.md")]
//~^ WARNING deprecated
//~| WARNING hard error in a future release

extern crate inner;

pub use inner::HasDocInclude;

#[doc(include = "auxiliary/docs.md")]
//~^ WARNING deprecated
//~| WARNING hard error in a future release
pub use inner::HasDocInclude as _;

#[doc(include = "auxiliary/docs.md")]
//~^ WARNING deprecated
//~| WARNING hard error in a future release
pub use inner::NoDocs;
