// run-pass
#![feature(box_syntax)]

fn f(i: Box<isize>) {
    assert_eq!(*i, 100);
}

pub fn main() {
    let i = box 100;
    f(i);
}
