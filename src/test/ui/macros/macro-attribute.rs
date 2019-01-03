#![feature(unrestricted_attribute_tokens)]

#[doc = $not_there] //~ ERROR unexpected token: `$`
fn main() { }
