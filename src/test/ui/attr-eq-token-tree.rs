// compile-pass

#![feature(custom_attribute, unrestricted_attribute_tokens)]

#[my_attr = !] // OK under feature gate
fn main() {}
