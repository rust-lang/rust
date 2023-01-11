// run-pass

#![deny(unused_parens)]
#![allow(unreachable_code)]

fn main() {
    match (return) {} // ok
    if (return) {} // ok
}
