// check-pass
#![feature(const_eval_limit)]
#![const_eval_limit="2"]

const CONSTANT: usize = limit();

fn main() {
    assert_eq!(CONSTANT, 1764);
}

const fn limit() -> usize {
    let x = 42;

    x * 42
}
