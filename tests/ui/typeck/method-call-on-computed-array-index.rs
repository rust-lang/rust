//! Regression test for <https://github.com/rust-lang/rust/issues/41604>.
//! This used to ICE.
//@ run-pass

struct B;

impl B {
    fn init(&mut self) {}
}

fn main() {
    let mut b = [B];
    b[1-1].init();
}
