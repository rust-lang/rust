//@ aux-build:t.rs
//@ build-aux-docs
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

//@ has t/trait.Tango.html
//@ hasraw search-index.js 'Quebec'
//@ has index.html '//ul[@class="all-items"]//a[@href="q/index.html"]' 'q'
//@ hasraw s/struct.Sierra.html 'Tango'
//@ hasraw trait.impl/t/trait.Tango.js 'struct.Sierra.html'
//@ has index.html '//ul[@class="all-items"]//a[@href="t/index.html"]' 't'
//@ has index.html '//h1' 'List of all crates'
//@ has index.html '//ul[@class="all-items"]//a[@href="s/index.html"]' 's'
//@ hasraw search-index.js 'Sierra'
//@ hasraw search-index.js 'Tango'
//@ has index.html
//@ has q/struct.Quebec.html
//@ has s/struct.Sierra.html

// We document multiple crates into the same output directory, which merges the cross-crate information. Everything is available.

extern crate t;
pub struct Sierra;
impl t::Tango for Sierra {}
