//@ aux-build:quebec.rs
//@ build-aux-docs
//@ doc-flags:--merge=none
//@ doc-flags:--parts-out-dir=info/doc.parts/tango
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

extern crate quebec;
pub trait Tango {}
