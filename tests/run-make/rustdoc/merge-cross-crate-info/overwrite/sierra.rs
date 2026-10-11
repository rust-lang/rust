//@ has quebec/type.Quebec.html
//@ has sierra/type.Sierra.html
//@ has tango/trait.Tango.html
//@ hasraw sierra/type.Sierra.html 'Tango'
//@ hasraw trait.impl/tango/trait.Tango.js 'type.Sierra.html'
//@ hasraw search.index/name/*.js 'Tango'
//@ hasraw search.index/name/*.js 'Sierra'
//@ !hasraw search.index/name/*.js 'Quebec'

// since tango is documented with "finalize mode", we overwrite q's
// cross-crate information
extern crate tango;
pub struct Sierra;
impl tango::Tango for Sierra {}
