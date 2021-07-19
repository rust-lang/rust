#![no_std]

// @has no_std/fn.foo.html '//a/[@href="{{channel}}/core/primitive.u8.html"]' 'u8'
// @has no_std/fn.foo.html '//a/[@href="{{channel}}/core/primitive.u8.html"]' 'primitive link'
/// Link to [primitive link][u8]
pub fn foo() -> u8 {}
