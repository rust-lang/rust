//@ aux-build:sierra.rs
//@ build-aux-docs
//@ unique-doc-out-dir
//@ doc-flags:--merge=none
//@ doc-flags:--parts-out-dir=info/doc.parts/romeo
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

extern crate sierra;
pub type Romeo = sierra::Sierra;
