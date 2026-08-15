//! Regression test for <https://github.com/rust-lang/rust/issues/17771>.
//! Tests that borrowing mut dyn trait field in a method doesn't cause ICE.

//@ run-pass
#![allow(dead_code)]

trait Aaa { fn dummy(&self) { } }

impl<'a> Aaa for &'a mut (dyn Aaa + 'a) {}

struct Bar<'a> {
    writer: &'a mut (dyn Aaa + 'a),
}

fn baz(_: &mut dyn Aaa) {
}

fn foo<'a>(mut bar: Bar<'a>) {
    baz(&mut bar.writer);
}

fn main() {
}
