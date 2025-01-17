//@ run-pass

use std::any::Any;

fn foo(_: &u8) {
}

fn main() {
    let _ = &foo as &dyn Any;
}
