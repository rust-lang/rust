//! Regression test for https://github.com/rust-lang/rust/issues/30891

//@ run-pass
const ERROR_CONST: bool = true;

fn get() -> bool {
    false || ERROR_CONST
}

pub fn main() {
    assert_eq!(get(), true);
}
