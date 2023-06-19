// aux-build:alias-reexport.rs
// aux-build:alias-reexport2.rs

#![crate_name = "foo"]
#![feature(lazy_type_alias)]

extern crate alias_reexport2;

// @has 'foo/reexport/fn.foo.html'
// FIXME: should be 'pub fn foo() -> Reexport'
// @has - '//*[@class="rust item-decl"]' 'pub fn foo() -> u8'
// @has 'foo/reexport/fn.foo2.html'
// FIXME: should be 'pub fn foo2() -> Result<Reexport, ()>'
// @has - '//*[@class="rust item-decl"]' 'pub fn foo2() -> Result<u8, ()>'
// @has 'foo/reexport/type.Reexported.html'
// @has - '//*[@class="rust item-decl"]' 'pub type Reexported = u8;'
#[doc(inline)]
pub use alias_reexport2 as reexport;
