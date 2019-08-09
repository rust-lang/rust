// run-pass
#![feature(box_syntax)]

trait Foo {
    fn f(self: Box<Self>);
}

struct S {
    x: isize
}

impl Foo for S {
    fn f(self: Box<S>) {
        assert_eq!(self.x, 3);
    }
}

pub fn main() {
    let x = box S { x: 3 };
    let y = x as Box<dyn Foo>;
    y.f();
}
