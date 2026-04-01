// https://github.com/rust-lang/rust/issues/12909
//@ run-pass
#![allow(unused_variables)]

use std::collections::HashMap;

fn copy<T: Copy>(&x: &T) -> T {
    x
}

fn main() {
    let arr = [(1, 1), (2, 2), (3, 3)];

    let v1: Vec<&_> = arr.iter().collect();
    let v2: Vec<_> = arr.iter().map(copy).collect();

    let m1: HashMap<_, _> = arr.iter().map(copy).collect();
    let m2: HashMap<isize, _> = arr.iter().map(copy).collect();
    let m3: HashMap<_, usize> = arr.iter().map(copy).collect();
}
