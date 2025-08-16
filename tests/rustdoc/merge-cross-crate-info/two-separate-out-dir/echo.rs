//@ aux-build:foxtrot.rs
//@ build-aux-docs
//@ doc-flags:--include-parts-dir=info/doc.parts/foxtrot
//@ doc-flags:-Zunstable-options

//@ has echo/enum.Echo.html
//@ hasraw echo/enum.Echo.html 'Foxtrot'
//@ hasraw trait.impl/foxtrot/trait.Foxtrot.js 'enum.Echo.html'
//@ hasraw search.index/name/*.js 'Foxtrot'
//@ hasraw search.index/name/*.js 'Echo'

// document two crates in different places, and merge their docs after
// they are generated
extern crate foxtrot;
pub enum Echo {}
impl foxtrot::Foxtrot for Echo {}
