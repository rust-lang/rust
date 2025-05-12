#![crate_name = "foo"]

//@ has foo/fn.f.html
//@ has - '//pre[@class="rust item-decl"]' 'pub fn f(callback: fn(len: usize, foo: u32))'
pub fn f(callback: fn(len: usize, foo: u32)) {}

//@ has foo/fn.g.html
//@ has - '//pre[@class="rust item-decl"]' 'pub fn g(_: fn(usize, u32))'
pub fn g(_: fn(usize, _: u32)) {}

//@ has foo/fn.mixed.html
//@ has - '//pre[@class="rust item-decl"]' 'pub fn mixed(_: fn(_: usize, foo: u32))'
pub fn mixed(_: fn(usize, foo: u32)) {}
