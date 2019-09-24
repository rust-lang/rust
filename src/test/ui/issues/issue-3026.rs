// run-pass
// pretty-expanded FIXME #23616

#![feature(box_syntax)]

use std::collections::HashMap;

pub fn main() {
    let x: Box<_>;
    let mut buggy_map: HashMap<usize, &usize> = HashMap::new();
    x = box 1;
    buggy_map.insert(42, &*x);
}
