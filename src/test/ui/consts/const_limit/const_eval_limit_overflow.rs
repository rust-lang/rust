#![feature(const_eval_limit)]
#![const_eval_limit="18_446_744_073_709_551_615"]
//~^ ERROR `limit` must be a non-negative integer

const CONSTANT: usize = limit();

fn main() {
    assert_eq!(CONSTANT, 1764);
}

const fn limit() -> usize {
    let x = 42;

    x * 42
}
