//@ has quebec/type.Quebec.html
//@ has sierra/type.Sierra.html
//@ has tango/trait.Tango.html
//@ hasraw sierra/type.Sierra.html 'Tango'
//@ hasraw trait.impl/tango/trait.Tango.js 'type.Sierra.html'
//@ hasraw search.index/name/*.js 'Tango'
//@ hasraw search.index/name/*.js 'Sierra'
//@ !hasraw search.index/name/*.js 'Quebec'

// we overwrite quebec and tango's cross-crate information, but we
// include the info from tango meaning that it should appear in the out
// dir
extern crate tango;
pub struct Sierra;
impl tango::Tango for Sierra {}
