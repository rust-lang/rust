#![deny(unknown_or_malformed_diagnostic_attributes)]

#[diagnostic::unknown_attribute]
//~^ERROR unknown diagnostic attribute
struct Foo;

fn main() {}
