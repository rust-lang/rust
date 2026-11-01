//@ aux-build:t.rs
//@ build-aux-docs
// simple test to see if we support building transitive crates
extern crate t;
pub struct Sierra;
impl t::Tango for Sierra {}
