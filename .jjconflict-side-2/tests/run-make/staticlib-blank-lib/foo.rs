#![crate_type = "staticlib"]

#[link(name = "foo", kind = "static")]
extern "C" {}

fn main() {}
