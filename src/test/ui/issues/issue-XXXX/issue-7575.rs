// run-pass

trait Foo {
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
