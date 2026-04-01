//@ aux-build:tango.rs
//@ build-aux-docs
//@ doc-flags:--merge=finalize
//@ doc-flags:--include-parts-dir=info/doc.parts/tango
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

//@ has quebec/struct.Quebec.html
//@ has sierra/struct.Sierra.html
//@ has tango/trait.Tango.html
//@ hasraw sierra/struct.Sierra.html 'Tango'
//@ hasraw trait.impl/tango/trait.Tango.js 'struct.Sierra.html'
//@ hasraw search.index/name/*.js 'Tango'
//@ hasraw search.index/name/*.js 'Sierra'
//@ !hasraw search.index/name/*.js 'Quebec'

// we overwrite quebec and tango's cross-crate information, but we
// include the info from tango meaning that it should appear in the out
// dir
extern crate tango;
pub struct Sierra;
impl tango::Tango for Sierra {}
