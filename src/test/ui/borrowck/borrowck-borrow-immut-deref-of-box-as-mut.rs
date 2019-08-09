#![feature(box_syntax)]

struct A;

impl A {
    fn foo(&mut self) {
    }
}

pub fn main() {
    let a: Box<_> = box A;
    a.foo();
    //~^ ERROR cannot borrow `*a` as mutable, as `a` is not declared as mutable [E0596]
}
