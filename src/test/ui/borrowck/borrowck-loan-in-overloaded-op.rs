#![feature(box_patterns)]
#![feature(box_syntax)]

use std::ops::Add;

#[derive(Clone)]
struct foo(Box<usize>);

impl Add for foo {
    type Output = foo;

    fn add(self, f: foo) -> foo {
        let foo(box i) = self;
        let foo(box j) = f;
        foo(box (i + j))
    }
}

fn main() {
    let x = foo(box 3);
    let _y = {x} + x.clone(); // the `{x}` forces a move to occur
    //~^ ERROR use of moved value: `x`
}
