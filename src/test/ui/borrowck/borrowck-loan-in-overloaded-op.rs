#![feature(box_patterns)]
#![feature(box_syntax)]

use std::ops::Add;

#[derive(Clone)]
struct Foo(Box<usize>);

impl Add for Foo {
    type Output = Foo;

    fn add(self, f: Foo) -> Foo {
        let Foo(box i) = self;
        let Foo(box j) = f;
        Foo(box (i + j))
    }
}

fn main() {
    let x = Foo(box 3);
    let _y = {x} + x.clone(); // the `{x}` forces a move to occur
    //~^ ERROR borrow of moved value: `x`
}
