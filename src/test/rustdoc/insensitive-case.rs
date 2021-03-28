// compile-flags: -Zunstable-options --generate-case-insensitive

#![crate_name = "foo"]

// @!has 'foo/struct.Aa.html'
// @has 'foo/struct.aa.html'
// @!has 'foo/struct.aa.html' '//h4[@id="method.new"]'
pub struct aa;

impl aa {
   pub fn foo(&self) {}
}

pub struct Aa;

impl Aa {
   pub fn foo(&self) {}
}
