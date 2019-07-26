// run-pass
#![feature(box_syntax)]

fn f(i: &mut Box<isize>) {
    *i = box 200;
}

pub fn main() {
    let mut i = box 100;
    f(&mut i);
    assert_eq!(*i, 200);
}
