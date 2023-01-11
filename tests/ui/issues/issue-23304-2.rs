// run-pass
#![allow(dead_code)]

enum X { A = 42 as isize }

enum Y { A = X::A as isize }

fn main() {
    let x = X::A;
    let x = x as isize;
    assert_eq!(x, 42);
    assert_eq!(Y::A as isize, 42);
}
