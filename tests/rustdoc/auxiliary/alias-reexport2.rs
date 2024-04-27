#![feature(lazy_type_alias)]

extern crate alias_reexport;

pub use alias_reexport::Reexported;

// @has 'foo/fn.foo.html'
// @has - '//*[@class="docblock item-decl"]' 'pub fn foo() -> Reexported'
pub fn foo() -> Reexported { 0 }
// @has 'foo/fn.foo2.html'
// @has - '//*[@class="docblock item-decl"]' 'pub fn foo2() -> Result<Reexported, ()>'
pub fn foo2() -> Result<Reexported, ()> { Ok(0) }
