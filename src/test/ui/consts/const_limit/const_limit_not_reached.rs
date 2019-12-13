// run-pass
#![feature(const_limit)]
#![const_limit="1000"]

const CONSTANT: usize = limit();

fn main() {
    assert_eq!(CONSTANT, 1764);
}

const fn limit() -> usize {
    let x = 42;

    x * 42
}
