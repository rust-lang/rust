// run-pass
#![feature(box_syntax)]

fn f<T>(t: T) -> T {
    let t1 = t;
    t1
}

pub fn main() {
    let t = f::<Box<_>>(box 100);
    assert_eq!(t, box 100);
}
