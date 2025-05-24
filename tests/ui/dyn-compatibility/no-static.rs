// Check that we correctly prevent users from making trait objects
// from traits with static methods.

trait Foo {
    fn foo() {}
}

fn diverges() -> Box<dyn Foo> {
    //~^ ERROR E0038
    loop { }
}

struct Bar;

impl Foo for Bar {}

fn main() {
    let b: Box<dyn Foo> = Box::new(Bar);
    //~^ ERROR E0038
}
