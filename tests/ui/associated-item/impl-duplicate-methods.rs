struct Foo;

impl Foo {
    fn orange(&self) {}
    fn orange(&self) {}
    //~^ ERROR duplicate definitions with name `orange` [E0592]
}

fn main() {}
