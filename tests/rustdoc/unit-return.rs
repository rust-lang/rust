// aux-build:unit-return.rs

#![crate_name = "foo"]

extern crate unit_return;

// @has 'foo/fn.f0.html' '//div[@class="item-decl"]/pre[@class="rust"]' 'F: FnMut(u8) + Clone'
pub fn f0<F: FnMut(u8) + Clone>(f: F) {}

// @has 'foo/fn.f1.html' '//div[@class="item-decl"]/pre[@class="rust"]' 'F: FnMut(u16) + Clone'
pub fn f1<F: FnMut(u16) -> () + Clone>(f: F) {}

// @has 'foo/fn.f2.html' '//div[@class="item-decl"]/pre[@class="rust"]' 'F: FnMut(u32) + Clone'
pub use unit_return::f2;

// @has 'foo/fn.f3.html' '//div[@class="item-decl"]/pre[@class="rust"]' 'F: FnMut(u64) + Clone'
pub use unit_return::f3;
