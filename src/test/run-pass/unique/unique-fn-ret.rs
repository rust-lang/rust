// run-pass
#![feature(box_syntax)]

fn f() -> Box<isize> {
    box 100
}

pub fn main() {
    assert_eq!(f(), box 100);
}
