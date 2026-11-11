//@ check-pass
#![allow(unused_assignments)]

#![allow(unused_variables)]

trait Foo {
    fn foo(&self, mut v: isize) { v = 1; }
}

pub fn main() {}
