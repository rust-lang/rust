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

// --write-doc-meta-dir on all crates does not write the search index
// in the actual output folder
extern crate tango;
pub struct Sierra;
impl tango::Tango for Sierra {}
