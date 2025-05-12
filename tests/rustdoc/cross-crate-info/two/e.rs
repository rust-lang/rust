//@ aux-build:f.rs
//@ build-aux-docs
// simple test to assert that we can do a two-level aux-build
extern crate f;
pub enum Echo {}
impl f::Foxtrot for Echo {}
