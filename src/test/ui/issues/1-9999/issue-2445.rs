// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

// pretty-expanded FIXME #23616

struct c1<T> {
    x: T,
}

impl<T> c1<T> {
    pub fn f1(&self, _x: T) {}
}

fn c1<T>(x: T) -> c1<T> {
    c1 {
        x: x
    }
}

impl<T> c1<T> {
    pub fn f2(&self, _x: T) {}
}


pub fn main() {
    c1::<isize>(3).f1(4);
    c1::<isize>(3).f2(4);
}
