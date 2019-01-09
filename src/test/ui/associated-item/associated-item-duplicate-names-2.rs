struct Foo;

impl Foo {
    const bar: bool = true;
    fn bar() {} //~ ERROR duplicate definitions
}

fn main() {}
