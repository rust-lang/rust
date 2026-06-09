//@ aux-build:tango.rs
//@ build-aux-docs
//@ unique-doc-out-dir
//@ doc-flags:--merge=none
//@ doc-flags:--parts-out-dir=info/doc.parts/sierra
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

extern crate tango;
pub struct Sierra;
impl tango::Tango for Sierra {}
