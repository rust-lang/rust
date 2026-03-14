#![feature(associated_type_defaults)]

trait Foo {
    // This causes cycle, as being effectively `Box<dyn Foo<Assoc = Box<dyn Foo<..>>>>`.
    type Assoc = Box<dyn Foo>;
    //~^ ERROR: cycle detected when computing type of `Foo::Assoc`
}

trait Bar {
    // This does not.
    type Assoc = Box<dyn Bar<Assoc = ()>>;
}

fn main() {}
