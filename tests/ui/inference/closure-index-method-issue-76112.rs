//! Regression test for https://github.com/rust-lang/rust/issues/76112.
//! A closure argument used as an index should be inferred from a later call to the closure.

//@ check-pass

use std::ops::Add;

struct Lhs;

impl Add<i32> for Lhs {
    type Output = i64;

    fn add(self, rhs: i32) -> Self::Output {
        rhs.into()
    }
}

fn main() {
    let array: [i64; 1] = [0];
    let get = |index| array[index].pow(1);

    let value: i64 = get(0);
    assert_eq!(value, 0);

    let options = [Some(1i32)];
    let map = |index| options[index].map(|value| value + 1);

    let value: Option<i32> = map(0);
    assert_eq!(value, Some(2));

    let get_chained = |index| array[index].wrapping_add(1).pow(1);
    let value: i64 = get_chained(0);
    assert_eq!(value, 1);

    let add = |rhs| (Lhs + rhs).pow(1);
    let value: i64 = add(2);
    assert_eq!(value, 2);
}
