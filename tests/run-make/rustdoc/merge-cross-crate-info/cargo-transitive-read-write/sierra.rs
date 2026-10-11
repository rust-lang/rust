//@ has index.html
//@ has index.html '//h1' 'List of all crates'
//@ has index.html '//ul[@class="all-items"]//a[@href="quebec/index.html"]' 'quebec'
//@ has index.html '//ul[@class="all-items"]//a[@href="sierra/index.html"]' 'sierra'
//@ has index.html '//ul[@class="all-items"]//a[@href="tango/index.html"]' 'tango'
//@ has help.html
//@ has settings.html
//@ has quebec/type.Quebec.html
//@ has sierra/type.Sierra.html
//@ has tango/trait.Tango.html
//@ hasraw sierra/type.Sierra.html 'Tango'
//@ hasraw trait.impl/tango/trait.Tango.js 'type.Sierra.html'
//@ hasraw search.index/name/*.js 'Tango'
//@ hasraw search.index/name/*.js 'Sierra'
//@ hasraw search.index/name/*.js 'Quebec'

// similar to cargo-workflow-transitive, but we use shared mode,
// which is the default.
extern crate tango;
pub struct Sierra;
impl tango::Tango for Sierra {}
