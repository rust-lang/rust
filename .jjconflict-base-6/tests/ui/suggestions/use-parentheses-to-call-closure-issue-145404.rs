//@ run-rustfix

use std::fmt::Display;

struct S;

impl S {
    fn call(&self, _: impl Display) {}
}

fn main() {
    S.call(|| "hello"); //~ ERROR [E0277]
}
