// compile-pass

#![feature(custom_attribute, unrestricted_attribute_tokens)]

#[my_attr(a b c d)]
fn main() {}
