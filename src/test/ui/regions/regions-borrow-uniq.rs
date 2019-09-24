// run-pass
#![feature(box_syntax)]

fn foo(x: &usize) -> usize {
    *x
}

pub fn main() {
    let p: Box<_> = box 3;
    let r = foo(&*p);
    assert_eq!(r, 3);
}
