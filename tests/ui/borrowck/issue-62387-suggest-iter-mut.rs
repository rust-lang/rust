//@ run-rustfix
#![allow(unused_mut)]
#![allow(dead_code)]

#[derive(Debug)]
struct A {
    a: i32,
}

impl A {
    fn double(&mut self) {
        self.a += self.a
    }
}

fn baz() {
    let mut v = [A { a: 4 }];
    v.iter().for_each(|a| a.double());
    //~^ ERROR cannot borrow `*a` as mutable, as it is behind a `&` reference
    println!("{:?}", v);
}

fn bar() {
    let mut v = [A { a: 4 }];
    v.iter().rev().rev().for_each(|a| a.double());
    //~^ ERROR cannot borrow `*a` as mutable, as it is behind a `&` reference
    println!("{:?}", v);
}

fn main() {}
