//@ aux-build:t.rs
//@ build-aux-docs
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

extern crate t;
pub struct Sierra;
impl t::Tango for Sierra {}
