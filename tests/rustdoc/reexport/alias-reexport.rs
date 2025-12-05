//@ aux-build:alias-reexport.rs
//@ aux-build:alias-reexport2.rs

#![crate_name = "foo"]
#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

extern crate alias_reexport2;

//@ has 'foo/reexport/fn.foo.html'
//@ has - '//*[@class="rust item-decl"]' 'pub fn foo() -> Reexported'
//@ has 'foo/reexport/fn.foo2.html'
//@ has - '//*[@class="rust item-decl"]' 'pub fn foo2() -> Result<Reexported, ()>'
//@ has 'foo/reexport/type.Reexported.html'
//@ has - '//*[@class="rust item-decl"]' 'pub type Reexported = u8;'
#[doc(inline)]
pub use alias_reexport2 as reexport;
