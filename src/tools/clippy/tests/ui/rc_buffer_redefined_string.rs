//@ check-pass

#![warn(clippy::rc_buffer)]

use std::rc::Rc;

struct String;

struct S {
    // does not trigger lint
    good1: Rc<String>,
}

fn main() {}
