//@ dont-require-annotations: NOTE

enum Thing {
    Foo(u8),
    Bar,
    Baz
}

fn main() {
    let Thing::Foo(y) = Thing::Foo(1);
    //~^ ERROR refutable pattern in local binding
    //~| NOTE `Thing::Bar` and `Thing::Baz` not covered
}
