// build-pass (FIXME(62277): could be check-pass?)

#![feature(existential_type)]

use std::fmt::Debug;

fn main() {
    existential type Existential: Debug;

    fn f() -> Existential {}
    println!("{:?}", f());
}
