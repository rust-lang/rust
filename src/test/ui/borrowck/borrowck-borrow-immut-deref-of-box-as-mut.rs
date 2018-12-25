#![feature(box_syntax)]

struct A;

impl A {
    fn foo(&mut self) {
    }
}

pub fn main() {
    let a: Box<_> = box A;
    a.foo();
    //~^ ERROR cannot borrow immutable `Box` content `*a` as mutable
}
