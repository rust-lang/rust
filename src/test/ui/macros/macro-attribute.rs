#![feature(unrestricted_attribute_tokens)]

#[doc = $not_there] //~ ERROR expected `]`, found `not_there`
fn main() { }
