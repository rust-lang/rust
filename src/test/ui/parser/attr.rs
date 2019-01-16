#![feature(lang_items)]

fn main() {}

#![lang = "foo"] //~ ERROR an inner attribute is not permitted in this context
                 //~| ERROR definition of an unknown language item: `foo`
fn foo() {}
