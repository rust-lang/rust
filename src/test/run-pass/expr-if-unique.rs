#![feature(box_syntax)]

// Tests for if as expressions returning boxed types
fn test_box() {
    let rs: Box<_> = if true { box 100 } else { box 101 };
    assert_eq!(*rs, 100);
}

pub fn main() { test_box(); }
