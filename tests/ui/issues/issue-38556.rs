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
