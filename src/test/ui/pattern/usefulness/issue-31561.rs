enum Thing {
    Foo(u8),
    Bar,
    Baz
}

fn main() {
    let Thing::Foo(y) = Thing::Foo(1);
    //~^ ERROR refutable pattern in local binding: `Thing::Bar` and `Thing::Baz` not covered
}
