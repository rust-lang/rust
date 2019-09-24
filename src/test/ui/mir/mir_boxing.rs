// run-pass
#![feature(box_syntax)]

fn test() -> Box<i32> {
    box 42
}

fn main() {
    assert_eq!(*test(), 42);
}
