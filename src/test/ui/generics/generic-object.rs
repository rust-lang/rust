// run-pass
#![feature(box_syntax)]

trait Foo<T> {
    fn get(&self) -> T;
}

struct S {
    x: isize
}

impl Foo<isize> for S {
    fn get(&self) -> isize {
        self.x
    }
}

pub fn main() {
    let x = box S { x: 1 };
    let y = x as Box<dyn Foo<isize>>;
    assert_eq!(y.get(), 1);
}
