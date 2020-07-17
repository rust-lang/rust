// run-pass

#![feature(const_option)]

const X: Option<i32> = Some(32);
const Y: Option<&i32> = X.as_ref();

const IS_SOME: bool = X.is_some();
const IS_NONE: bool = Y.is_none();

fn main() {
    assert!(IS_SOME);
    assert!(!IS_NONE)
}
