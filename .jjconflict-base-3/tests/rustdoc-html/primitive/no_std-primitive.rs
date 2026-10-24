#![no_std]

/// Link to [intra-doc link][u8]
//@ has 'no_std_primitive/fn.foo.html' '//a[@href="{{channel}}/core/primitive.u8.html"]' 'intra-doc link'
//@ has - '//a[@href="{{channel}}/core/primitive.u8.html"]' 'u8'
pub fn foo() -> u8 {}
