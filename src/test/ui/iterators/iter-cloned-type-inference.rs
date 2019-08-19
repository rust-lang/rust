// run-pass
#![allow(stable_features)]

// Test to see that the element type of .cloned() can be inferred
// properly. Previously this would fail to deduce the type of `sum`.

#![feature(iter_arith)]

fn square_sum(v: &[i64]) -> i64 {
    let sum: i64 = v.iter().cloned().sum();
    sum * sum
}

fn main() {
    assert_eq!(36, square_sum(&[1,2,3]));
}
