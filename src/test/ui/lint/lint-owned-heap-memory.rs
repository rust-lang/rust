#![allow(dead_code)]
#![forbid(box_pointers)]
#![feature(box_syntax)]

struct Foo {
    x: Box<isize> //~ ERROR type uses owned
}

fn main() {
    let _x : Foo = Foo {x : box 10};
    //~^ ERROR type uses owned
}
