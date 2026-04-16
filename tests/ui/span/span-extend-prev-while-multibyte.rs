//@ build-pass
// Regression test for https://github.com/rust-lang/rust/issues/155037
#![feature(associated_type_defaults)]

trait Trait {
    type 否 where = ();
    //~^ WARNING where clause not allowed here
}

fn main() {}
