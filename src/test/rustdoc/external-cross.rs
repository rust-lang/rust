// aux-build:external-cross.rs
// ignore-cross-compile

#![crate_name="host"]

extern crate external_cross;

// @has host/struct.NeedMoreDocs.html
// @has - '//h1' 'Cross-crate imported docs'
pub use external_cross::NeedMoreDocs;
