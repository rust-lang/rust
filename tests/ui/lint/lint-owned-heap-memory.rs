#![allow(dead_code)]
#![forbid(box_pointers)]


struct Foo {
    x: Box<isize> //~ ERROR type uses owned
}

fn main() {
    let _x: Foo = Foo { x : Box::new(10) };
    //~^ ERROR type uses owned
}
