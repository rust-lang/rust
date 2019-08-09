// build-pass (FIXME(62277): could be check-pass?)

#![deny(unused_mut)]

pub fn foo() {
    return;

    let mut v = 0;
    assert_eq!(v, 0);
    v = 1;
    assert_eq!(v, 1);
}

fn main() {}
