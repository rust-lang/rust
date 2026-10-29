//! Auxiliary crate testing this issue https://github.com/rust-lang/rust/issues/2414
#![crate_name="a"]
#![crate_type = "lib"]

type t1 = usize;

trait foo {
    fn foo(&self);
}

impl foo for String {
    fn foo(&self) {}
}
