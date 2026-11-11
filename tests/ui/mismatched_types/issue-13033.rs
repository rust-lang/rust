//@ dont-require-annotations: NOTE

trait Foo {
    fn bar(&mut self, other: &mut dyn Foo);
}

struct Baz;

impl Foo for Baz {
    fn bar(&mut self, other: &dyn Foo) {}
    //~^ ERROR method `bar` has an incompatible type for trait
    //~| NOTE expected signature `fn(&mut Baz, &mut dyn Foo)`
    //~| NOTE found signature `fn(&mut Baz, &dyn Foo)`
}

fn main() {}
