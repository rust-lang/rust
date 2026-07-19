//@ aux-build:quebec.rs
//@ build-aux-docs
//@ unique-doc-out-dir
//@ doc-flags:--write-doc-meta-dir=info/doc.parts/tango
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

extern crate quebec;
pub trait Tango {}
