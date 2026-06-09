//@ run-pass

use std::collections::HashMap;

pub fn main() {
    let mut h = HashMap::new();
    let kvs = [(1, 10), (2, 20), (3, 30)];
    for &(k,v) in &kvs {
        h.insert(k,v);
    }
    let mut x = 0;
    let mut y = 0;
    for (&k,&v) in &h {
        x += k;
        y += v;
    }
    assert_eq!(x, 6);
    assert_eq!(y, 60);
}
