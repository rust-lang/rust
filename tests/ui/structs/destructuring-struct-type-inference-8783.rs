// https://github.com/rust-lang/rust/issues/8783
//@ run-pass
#![allow(unused_variables)]

struct X { pub x: usize }
impl Default for X {
    fn default() -> X {
        X { x: 42 }
    }
}

struct Y<T> { pub y: T }
impl<T: Default> Default for Y<T> {
    fn default() -> Y<T> {
        Y { y: Default::default() }
    }
}

fn main() {
    let X { x: _ } = Default::default();
    let Y { y: X { x } } = Default::default();
}
