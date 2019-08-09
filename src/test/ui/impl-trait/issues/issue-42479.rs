// build-pass (FIXME(62277): could be check-pass?)

use std::iter::once;

struct Foo {
    x: i32,
}

impl Foo {
    fn inside(&self) -> impl Iterator<Item = &i32> {
        once(&self.x)
    }
}

fn main() {
    println!("hi");
}
