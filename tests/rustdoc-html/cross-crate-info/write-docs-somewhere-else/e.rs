//@ aux-build:f.rs
//@ build-aux-docs
//@ has e/type.Echo.html
//@ !has f/trait.Foxtrot.html
//@ hasraw e/type.Echo.html 'Foxtrot'
//@ hasraw trait.impl/f/trait.Foxtrot.js 'type.Echo.html'
//@ !hasraw search.index/name/*.js 'Foxtrot'
//@ hasraw search.index/name/*.js 'Echo'

// test the fact that our test runner will document this crate somewhere
// else
extern crate f;
pub enum Echo {}
impl f::Foxtrot for Echo {}
