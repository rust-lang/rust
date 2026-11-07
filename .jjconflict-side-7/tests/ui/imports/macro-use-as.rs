//! Regression test for <https://github.com/rust-lang/rust/issues/38556>.
//! Reexport in macro caused ICE.
//@ run-pass

#![allow(dead_code)]
pub struct Foo;

macro_rules! reexport {
    () => { use Foo as Bar; }
}

reexport!();

fn main() {
    fn f(_: Bar) {}
}
