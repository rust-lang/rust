//@ aux-build:pointer-reexports-allowed.rs
//@ check-pass
extern crate inner;
pub use inner::foo;
