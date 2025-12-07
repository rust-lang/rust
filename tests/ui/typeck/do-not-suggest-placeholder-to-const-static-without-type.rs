trait Foo {
    const A; //~ ERROR omitting type on const item declaration is experimental [E0658]
    static B;
    //~^ ERROR associated `static` items are not allowed
    //~| ERROR missing type for `static` item
}

fn main() {}
