//@ aux-build:t.rs
//@ build-aux-docs
//@ has q/struct.Quebec.html
//@ has s/struct.Sierra.html
//@ has t/trait.Tango.html
//@ hasraw s/struct.Sierra.html 'Tango'
//@ hasraw trait.impl/t/trait.Tango.js 'struct.Sierra.html'
//@ hasraw search.index/name/*.js 'Tango'
//@ hasraw search.index/name/*.js 'Sierra'
//@ hasraw search.index/name/*.js 'Quebec'

// We document multiple crates into the same output directory, which
// merges the cross-crate information. Everything is available.
extern crate t;
pub struct Sierra;
impl t::Tango for Sierra {}
