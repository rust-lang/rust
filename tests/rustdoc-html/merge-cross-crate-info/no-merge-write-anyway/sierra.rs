//@ aux-build:tango.rs
//@ build-aux-docs
//@ doc-flags:--merge=none
//@ doc-flags:--parts-out-dir=info/doc.parts/sierra
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

//@ !has index.html
//@ has sierra/struct.Sierra.html
//@ has tango/trait.Tango.html
//@ hasraw sierra/struct.Sierra.html 'Tango'
//@ !has trait.impl/tango/trait.Tango.js
//@ !has search.index/name/*.js

// we --merge=none, so --parts-out-dir doesn't do anything
extern crate tango;
pub struct Sierra;
impl tango::Tango for Sierra {}
