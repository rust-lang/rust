struct Foo;

impl Foo {
    fn orange(&self) {}
    fn orange(&self) {}
    //~^ ERROR duplicate definition
}

fn main() {}
