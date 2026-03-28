// Check that specialization must be ungated to use the `default` keyword

trait Foo {
    fn foo(&self);
}

default impl<T> Foo for T { //~ ERROR specialization is experimental
    fn foo(&self) {}
}

fn main() {}
