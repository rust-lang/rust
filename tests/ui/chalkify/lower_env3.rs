// check-pass
// compile-flags: -Z trait-solver=chalk

#![allow(dead_code)]

trait Foo {
    fn foo(&self);
}

impl<T> Foo for T where T: Clone {
    fn foo(&self) {
    }
}

fn main() {
}
