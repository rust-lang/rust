//@ !has index.html
//@ !has help.html
//@ !has settings.html
//@ has sierra/type.Sierra.html
//@ has tango/trait.Tango.html
//@ hasraw sierra/type.Sierra.html 'Tango'
//@ !has trait.impl/tango/trait.Tango.js
//@ !has search.index/name/*.js

// we don't use --read-doc-meta-dir, so no metadata is loaded
extern crate tango;
pub struct Sierra;
impl tango::Tango for Sierra {}
