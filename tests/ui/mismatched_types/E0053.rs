trait Foo {
    fn foo(x: u16);
    fn bar(&self);
}

struct Bar;

impl Foo for Bar {
    fn foo(x: i16) { }
    //~^ ERROR method `foo` has an incompatible type for trait
    fn bar(&mut self) { }
    //~^ ERROR method `bar` has an incompatible type for trait
}

fn main() {
}
