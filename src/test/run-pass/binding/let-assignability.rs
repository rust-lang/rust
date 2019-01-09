// run-pass
#![feature(box_syntax)]

fn f() {
    let a: Box<_> = box 1;
    let b: &isize = &*a;
    println!("{}", b);
}

pub fn main() {
    f();
}
