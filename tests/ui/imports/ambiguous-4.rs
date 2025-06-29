//@ ignore-test
// ^^ don't know how to change this test
//@ aux-build: ../ambiguous-4-extern.rs

extern crate ambiguous_4_extern;

fn main() {
    ambiguous_4_extern::id();
    //^ FIXME: `id` should be identified as an ambiguous item.
}
