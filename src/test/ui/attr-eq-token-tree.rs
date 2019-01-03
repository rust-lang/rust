#![feature(custom_attribute, unrestricted_attribute_tokens)]

#[my_attr = !] //~ ERROR unexpected token: `!`
fn main() {}
