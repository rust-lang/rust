// aux-build:plugin.rs
// ignore-stage1

#[macro_use] extern crate plugin;

#[derive(Foo, Bar)] //~ ERROR proc-macro derive panicked
struct Baz {
    a: i32,
    b: i32,
}

fn main() {}
