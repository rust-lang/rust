// pretty-expanded FIXME #23616

#![allow(dead_assignment)]
#![allow(unused_variables)]

trait Foo {
    fn foo(&self, mut v: isize) { v = 1; }
}

pub fn main() {}
