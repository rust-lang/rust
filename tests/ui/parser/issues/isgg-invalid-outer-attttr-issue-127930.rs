#![allow(dead_code)]
fn foo() {}

#![feature(iter_array_chunks)] //~ ERROR an inner attribute is not permitted in this context
fn bar() {}

fn main() {
    foo();
    bar();
}
