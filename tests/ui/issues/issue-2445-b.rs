//@ run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]


struct c1<T> {
    x: T,
}

impl<T> c1<T> {
    pub fn f1(&self, _x: isize) {
    }
}

fn c1<T>(x: T) -> c1<T> {
    c1 {
        x: x
    }
}

impl<T> c1<T> {
    pub fn f2(&self, _x: isize) {
    }
}


pub fn main() {
    c1::<isize>(3).f1(4);
    c1::<isize>(3).f2(4);
}
