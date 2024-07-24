//@ aux-build:f.rs
//@ build-aux-docs
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

//@ hasraw search-index.js 'Echo'
//@ hasraw search-index.js 'Foxtrot'
//@ has index.html '//ul[@class="all-items"]//a[@href="e/index.html"]' 'e'
//@ has index.html '//ul[@class="all-items"]//a[@href="f/index.html"]' 'f'
//@ hasraw trait.impl/f/trait.Foxtrot.js 'enum.Echo.html'
//@ has f/trait.Foxtrot.html
//@ has index.html '//h1' 'List of all crates'
//@ has e/enum.Echo.html
//@ has index.html
//@ hasraw e/enum.Echo.html 'Foxtrot'

// document two crates in the same way that cargo does, writing them both into the same output directory

extern crate f;
pub enum Echo {}
impl f::Foxtrot for Echo {}
