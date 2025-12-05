//@ run-pass
#![allow(dead_code)]
const ARR: [usize; 1] = [2];
const ARR2: [i32; ARR[0]] = [5, 6];

fn main() {
}
