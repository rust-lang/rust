// run-pass
#![feature(box_syntax)]

pub fn main() {
    let bar: Box<_> = box 3;
    let h = || -> isize { *bar };
    assert_eq!(h(), 3);
}
