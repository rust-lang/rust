//@ aux-build:f.rs
//@ build-aux-docs
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

//@ has index.html
//@ has index.html '//h1' 'List of all crates'
//@ has index.html '//ul[@class="all-items"]//a[@href="f/index.html"]' 'f'
//@ has index.html '//ul[@class="all-items"]//a[@href="e/index.html"]' 'e'
//@ has e/enum.Echo.html
//@ has f/trait.Foxtrot.html
//@ hasraw e/enum.Echo.html 'Foxtrot'
//@ hasraw trait.impl/f/trait.Foxtrot.js 'enum.Echo.html'
//@ hasraw search-index.js 'Foxtrot'
//@ hasraw search-index.js 'Echo'

// only declare --enable-index-page to the last rustdoc invocation
extern crate f;
pub enum Echo {}
impl f::Foxtrot for Echo {}
