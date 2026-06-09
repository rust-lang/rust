use std::fmt::Display;

trait Foo {
    fn bar(&self) -> impl Display;
}

impl Foo for () {
    fn bar(&self) -> impl Display {
        "Hello, world"
    }
}

fn main() {
    let x: &str = ().bar();
    //~^ ERROR mismatched types
}
