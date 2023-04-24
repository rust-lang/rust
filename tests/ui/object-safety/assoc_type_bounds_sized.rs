trait Foo {
    type Bar
    where
        Self: Sized;
}

fn foo(_: &dyn Foo) {} //~ ERROR: the value of the associated type `Bar` (from trait `Foo`) must be specified

fn main() {}
