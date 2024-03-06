//@ run-pass
#![allow(unused_variables)]
fn foo((x, y): (i8, i8)) {
}

fn main() {
    foo((0, 1));
}
