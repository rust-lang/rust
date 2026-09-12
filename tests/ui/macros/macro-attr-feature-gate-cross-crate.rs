//@ aux-build:macro_attr_feature_gate.rs
#![crate_type = "lib"]

extern crate macro_attr_feature_gate as foo;

#[foo::identity]
//~^ ERROR `macro_rules!` attributes are unstable
fn main() {}
