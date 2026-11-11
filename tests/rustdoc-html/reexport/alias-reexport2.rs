// gate-test-checked_type_aliases
//@ aux-build:alias-reexport.rs

#![crate_name = "foo"]
#![feature(checked_type_aliases)]
#![allow(incomplete_features)]

extern crate alias_reexport;

use alias_reexport::Reexported;

//@ has 'foo/fn.foo.html'
//@ has - '//*[@class="rust item-decl"]' 'pub fn foo() -> Reexported'
pub fn foo() -> Reexported { 0 }
//@ has 'foo/fn.foo2.html'
//@ has - '//*[@class="rust item-decl"]' 'pub fn foo2() -> Result<Reexported, ()>'
pub fn foo2() -> Result<Reexported, ()> { Ok(0) }
