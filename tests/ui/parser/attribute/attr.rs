#![feature(lang_items)]

fn main() {}

#![lang = "foo"] //~ ERROR an inner attribute is not permitted in this context
fn foo() {}
