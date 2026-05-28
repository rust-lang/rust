//@ aux-build:f.rs
//@ build-aux-docs
//@ has e/enum.Echo.html
//@ has f/trait.Foxtrot.html
//@ hasraw e/enum.Echo.html 'Foxtrot'
//@ hasraw trait.impl/f/trait.Foxtrot.js 'enum.Echo.html'
//@ hasraw search.index/name/*.js 'Foxtrot'
//@ hasraw search.index/name/*.js 'Echo'

// document two crates in the same way that cargo does. do not provide
// --enable-index-page
extern crate f;
pub enum Echo {}
impl f::Foxtrot for Echo {}
