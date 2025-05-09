//@ aux-build:tango.rs
//@ build-aux-docs
//@ doc-flags:--merge=shared
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

//@ has index.html
//@ has index.html '//h1' 'List of all crates'
//@ has index.html '//ul[@class="all-items"]//a[@href="quebec/index.html"]' 'quebec'
//@ has index.html '//ul[@class="all-items"]//a[@href="sierra/index.html"]' 'sierra'
//@ has index.html '//ul[@class="all-items"]//a[@href="tango/index.html"]' 'tango'
//@ has quebec/struct.Quebec.html
//@ has sierra/struct.Sierra.html
//@ has tango/trait.Tango.html
//@ hasraw sierra/struct.Sierra.html 'Tango'
//@ hasraw trait.impl/tango/trait.Tango.js 'struct.Sierra.html'
//@ hasraw search.index/name/*.js 'Tango'
//@ hasraw search.index/name/*.js 'Sierra'
//@ hasraw search.index/name/*.js 'Quebec'

// We can use read-write to emulate the default behavior of rustdoc, when
// --merge is left out.
extern crate tango;
pub struct Sierra;
impl tango::Tango for Sierra {}
