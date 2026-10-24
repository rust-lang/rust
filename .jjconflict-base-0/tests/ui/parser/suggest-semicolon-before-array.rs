//@ run-rustfix
#![allow(dead_code)]

fn foo() {}

fn bar() -> [u8; 2] {
    foo()
    [1, 3] //~ ERROR expected `;`, found `[`
}

fn main() {}
