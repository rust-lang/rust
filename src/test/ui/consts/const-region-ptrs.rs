// run-pass
#![allow(non_upper_case_globals)]

struct Pair<'a> { a: isize, b: &'a isize }

const x: &'static isize = &10;

const y: &'static Pair<'static> = &Pair {a: 15, b: x};

pub fn main() {
    println!("x = {}", *x);
    println!("y = {{a: {}, b: {}}}", y.a, *(y.b));
    assert_eq!(*x, 10);
    assert_eq!(*(y.b), 10);
}
