// xfail-fast
// aux-build:explicit_self_xcrate.rs

extern mod explicit_self_xcrate;
use explicit_self_xcrate::{Foo, Bar};

fn main() {
    let x = Bar { x: ~"hello" };
    x.f();
}

