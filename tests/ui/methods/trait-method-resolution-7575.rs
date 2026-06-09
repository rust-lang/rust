// https://github.com/rust-lang/rust/issues/7575
//@ run-pass

trait Foo { //~ WARN trait `Foo` is never used
    fn new() -> bool { false }
    fn dummy(&self) { }
}

trait Bar {
    fn new(&self) -> bool { true }
}

impl Bar for isize {}
impl Foo for isize {}

fn main() {
    assert!(1.new());
}
