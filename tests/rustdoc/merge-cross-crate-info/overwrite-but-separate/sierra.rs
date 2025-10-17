//@ aux-build:tango.rs
//@ build-aux-docs
//@ doc-flags:--merge=finalize
//@ doc-flags:--include-parts-dir=info/doc.parts/tango
//@ doc-flags:--include-parts-dir=info/doc.parts/quebec
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

//@ has index.html
//@ has index.html '//h1' 'List of all crates'
//@ has index.html '//ul[@class="all-items"]//a[@href="quebec/index.html"]' 'quebec'
//@ has index.html '//ul[@class="all-items"]//a[@href="sierra/index.html"]' 'sierra'
//@ has index.html '//ul[@class="all-items"]//a[@href="tango/index.html"]' 'tango'
//@ has sierra/struct.Sierra.html
//@ hasraw trait.impl/tango/trait.Tango.js 'struct.Sierra.html'
//@ hasraw search.index/name/*.js 'Tango'
//@ hasraw search.index/name/*.js 'Sierra'
//@ hasraw search.index/name/*.js 'Quebec'

// If these were documeted into the same directory, the info would be
// overwritten. However, since they are merged, we can still recover all
// of the cross-crate information
extern crate tango;
pub struct Sierra;
impl tango::Tango for Sierra {}
