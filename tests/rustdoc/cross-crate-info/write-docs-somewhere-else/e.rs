//@ aux-build:f.rs
//@ build-aux-docs
//@ has e/enum.Echo.html
//@ !has f/trait.Foxtrot.html
//@ hasraw e/enum.Echo.html 'Foxtrot'
//@ hasraw trait.impl/f/trait.Foxtrot.js 'enum.Echo.html'
//@ !hasraw search.index/name/*.js 'Foxtrot'
//@ hasraw search.index/name/*.js 'Echo'

// test the fact that our test runner will document this crate somewhere
// else
extern crate f;
pub enum Echo {}
impl f::Foxtrot for Echo {}
