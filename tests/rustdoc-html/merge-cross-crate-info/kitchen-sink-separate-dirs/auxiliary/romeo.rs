//@ aux-build:sierra.rs
//@ build-aux-docs
//@ unique-doc-out-dir
//@ doc-flags:--write-doc-meta-dir=info/doc.parts/romeo
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

extern crate sierra;
pub type Romeo = sierra::Sierra;
