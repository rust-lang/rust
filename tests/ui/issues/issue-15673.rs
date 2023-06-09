// run-pass
#![allow(stable_features)]

#![feature(iter_arith)]

fn main() {
    let x: [u64; 3] = [1, 2, 3];
    assert_eq!(6, (0..3).map(|i| x[i]).sum::<u64>());
}
