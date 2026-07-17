//@ aux-build:tango.rs
//@ build-aux-docs
//@ doc-flags:--write-doc-meta-dir=info/doc.parts/sierra
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

//@ !has index.html
//@ has sierra/struct.Sierra.html
//@ has tango/trait.Tango.html
//@ hasraw sierra/struct.Sierra.html 'Tango'
//@ !has trait.impl/tango/trait.Tango.js
//@ !has search.index/name/*.js

// we don't use --read-doc-meta-dir, so no metadata is loaded
extern crate tango;
pub struct Sierra;
impl tango::Tango for Sierra {}
