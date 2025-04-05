trait Foo {
    fn bar(&mut self, other: &mut dyn Foo);
}

struct Baz;

impl Foo for Baz {
    fn bar(&mut self, other: &dyn Foo) {}
    //~^ ERROR method `bar` has an incompatible type for trait
    //~| NOTE_NONVIRAL expected signature `fn(&mut Baz, &mut dyn Foo)`
    //~| NOTE_NONVIRAL found signature `fn(&mut Baz, &dyn Foo)`
}

fn main() {}
