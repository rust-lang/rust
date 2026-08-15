enum Foo {
    enum Bar { Baz },
    //~^ ERROR `enum` definition cannot be nested inside `enum`
    struct Quux { field: u8 },
    //~^ ERROR `struct` definition cannot be nested inside `enum`
    union Wibble { field: u8 },
    //~^ ERROR `union` definition cannot be nested inside `enum`
    Bat,
}

fn main() { }
