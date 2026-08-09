//! Regression test for https://github.com/rust-lang/rust/issues/63702.
//! An index type used in a closure body should be inferred from a later closure call.

//@ check-pass

fn choose(v: &Vec<(i32, i32)>) -> i32 {
    let x: usize = 0;
    let condition = |idx| v[idx].0 > 0;

    if condition(x) { 0 } else { 1 }
}

fn main() {
    assert_eq!(choose(&vec![(1, 2)]), 0);
}
