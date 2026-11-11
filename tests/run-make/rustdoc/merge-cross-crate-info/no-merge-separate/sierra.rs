//@ !has index.html
//@ !has help.html
//@ !has settings.html
//@ has sierra/struct.Sierra.html
//@ hasraw sierra/struct.Sierra.html 'Tango'
//@ !has trait.impl/tango/trait.Tango.js
//@ !has search.index/name/*.js

// we don't generate any cross-crate info if --write-doc-meta-dir, even if we
// document crates separately
extern crate tango;
pub struct Sierra;
impl tango::Tango for Sierra {}
