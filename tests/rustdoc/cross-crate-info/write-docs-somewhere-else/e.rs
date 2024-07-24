//@ aux-build:f.rs
//@ build-aux-docs

//@ hasraw search-index.js 'Echo'
//@ !hasraw search-index.js 'Foxtrot'
//@ hasraw trait.impl/f/trait.Foxtrot.js 'enum.Echo.html'
//@ !has f/trait.Foxtrot.html
//@ has e/enum.Echo.html
//@ hasraw e/enum.Echo.html 'Foxtrot'

// test the fact that our test runner will document this crate somewhere else

extern crate f;
pub enum Echo {}
impl f::Foxtrot for Echo {}
