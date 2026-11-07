//! Regression test for https://github.com/rust-lang/rust/issues/15104

//@ run-pass

fn main() {
    assert_eq!(count_members(&[1, 2, 3, 4]), 4);
}

fn count_members(v: &[usize]) -> usize {
    match *v {
        []         => 0,
        [_]        => 1,
        [_, ref xs @ ..] => 1 + count_members(xs)
    }
}
