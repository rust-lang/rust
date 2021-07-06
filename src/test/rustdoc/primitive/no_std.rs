#![no_std]

// @has no_std/fn.foo.html '//a/[@href="{{channel}}/core/primitive.u8.html"]' 'u8'
// Link to [u8]
pub fn foo() -> u8 {}
