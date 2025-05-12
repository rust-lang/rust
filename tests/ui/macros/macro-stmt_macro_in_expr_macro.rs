//@ run-pass
#![allow(dead_code)]
macro_rules! foo {
    () => {
        struct Bar;
        struct Baz;
    }
}

macro_rules! grault {
    () => {{
        foo!();
        struct Xyzzy;
        0
    }}
}

fn main() {
    let x = grault!();
    assert_eq!(x, 0);
}
