// gate-test-lazy_type_alias
// aux-build:alias-reexport.rs

#![crate_name = "foo"]
#![feature(lazy_type_alias)]

extern crate alias_reexport;

use alias_reexport::Reexported;

// @has 'foo/fn.foo.html'
// @has - '//*[@class="rust item-decl"]' 'pub fn foo() -> Reexported'
pub fn foo() -> Reexported { 0 }
// @has 'foo/fn.foo2.html'
// @has - '//*[@class="rust item-decl"]' 'pub fn foo2() -> Result<Reexported, ()>'
pub fn foo2() -> Result<Reexported, ()> { Ok(0) }
