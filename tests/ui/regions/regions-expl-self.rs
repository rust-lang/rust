// run-pass
#![allow(dead_code)]
// Test that you can insert an explicit lifetime in explicit self.

// pretty-expanded FIXME #23616

struct Foo {
    f: usize
}

impl Foo {
    pub fn foo<'a>(&'a self) {}
}

pub fn main() {}
