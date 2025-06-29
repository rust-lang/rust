//@ ignore-test
// ^^ don't know how to change this test
//@ aux-build: ../ambiguous-1.rs
// https://github.com/rust-lang/rust/pull/113099#issuecomment-1633574396

extern crate ambiguous_1;

fn main() {
    ambiguous_1::id();
    //^ FIXME: `id` should be identified as an ambiguous item.
}
