// run-pass

use std::collections::HashMap;

// This is a fancy one: it uses an external iterator established
// outside the loop, breaks, then _picks back up_ and continues
// iterating with it.

pub fn main() {
    let mut h = HashMap::new();
    let kvs = [(1, 10), (2, 20), (3, 30)];
    for &(k,v) in &kvs {
        h.insert(k,v);
    }
    let mut x = 0;
    let mut y = 0;

    let mut i = h.iter();

    for (&k,&v) in i.by_ref() {
        x += k;
        y += v;
        break;
    }

    for (&k,&v) in i {
        x += k;
        y += v;
    }

    assert_eq!(x, 6);
    assert_eq!(y, 60);
}
