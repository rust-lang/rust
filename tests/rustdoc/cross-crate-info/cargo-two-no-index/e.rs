//@ aux-build:f.rs
//@ build-aux-docs

//@ hasraw search-index.js 'Echo'
//@ hasraw search-index.js 'Foxtrot'
//@ hasraw trait.impl/f/trait.Foxtrot.js 'enum.Echo.html'
//@ has f/trait.Foxtrot.html
//@ has e/enum.Echo.html
//@ hasraw e/enum.Echo.html 'Foxtrot'

// document two crates in the same way that cargo does. do not provide --enable-index-page

extern crate f;
pub enum Echo {}
impl f::Foxtrot for Echo {}
