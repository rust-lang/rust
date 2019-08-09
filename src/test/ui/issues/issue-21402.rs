// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// pretty-expanded FIXME #23616

#[derive(Hash)]
struct Foo {
    a: Vec<bool>,
    b: (bool, bool),
    c: [bool; 2],
}

fn main() {}
