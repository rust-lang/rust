// run-pass
#![feature(const_limit)]
#![const_limit="18_446_744_073_709_551_615"]

const CONSTANT: usize = limit();

fn main() {
    assert_eq!(CONSTANT, 1764);
}

const fn limit() -> usize {
    let x = 42;

    x * 42
}
