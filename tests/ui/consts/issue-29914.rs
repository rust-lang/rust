//@ run-pass
#![allow(stable_features)]

#![feature(const_indexing)]

const ARR: [usize; 5] = [5, 4, 3, 2, 1];

fn main() {
    assert_eq!(3, ARR[ARR[3]]);
}
