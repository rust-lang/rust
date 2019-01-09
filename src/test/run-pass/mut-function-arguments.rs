#![feature(box_syntax)]

fn f(mut y: Box<isize>) {
    *y = 5;
    assert_eq!(*y, 5);
}

fn g() {
    let frob = |mut q: Box<isize>| { *q = 2; assert_eq!(*q, 2); };
    let w = box 37;
    frob(w);

}

pub fn main() {
    let z = box 17;
    f(z);
    g();
}
