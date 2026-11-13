//! Regression test for https://github.com/rust-lang/rust/issues/3026

//@ run-pass

use std::collections::HashMap;

pub fn main() {
    let x: Box<_>;
    let mut buggy_map: HashMap<usize, &usize> = HashMap::new();
    x = Box::new(1);
    buggy_map.insert(42, &*x);
}
