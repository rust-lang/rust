//! Regression test for <https://github.com/rust-lang/rust/issues/20186>.

//@ check-pass
#![allow(dead_code)]
#![allow(unused_variables)]
struct Foo;

impl Foo {
    fn putc(&self, b: u8) { }

    fn puts(&self, s: &str) {
        for byte in s.bytes() {
            self.putc(byte)
        }
    }
}

fn main() {}
