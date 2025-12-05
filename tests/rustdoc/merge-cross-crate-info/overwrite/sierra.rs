//@ aux-build:tango.rs
//@ build-aux-docs
//@ doc-flags:--merge=shared
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

// since tango is documented with --merge=finalize, we overwrite q's
// cross-crate information
extern crate tango;
pub struct Sierra;
impl tango::Tango for Sierra {}
