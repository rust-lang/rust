// run-pass

#![allow(dead_code)]
// pretty-expanded FIXME #23616

struct S<T> {
    contents: T,
}

impl<T> S<T> {
    fn new<U>(x: T, _: U) -> S<T> {
        S {
            contents: x,
        }
    }
}

trait Trait<T> {
    fn new<U>(x: T, y: U) -> Self;
}

struct S2 {
    contents: isize,
}

impl Trait<isize> for S2 {
    fn new<U>(x: isize, _: U) -> S2 {
        S2 {
            contents: x,
        }
    }
}

pub fn main() {
    let _ = S::<isize>::new::<f64>(1, 1.0);
    let _: S2 = Trait::<isize>::new::<f64>(1, 1.0);
}
