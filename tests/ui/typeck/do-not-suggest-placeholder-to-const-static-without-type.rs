trait Foo {
    const A; //~ ERROR missing type for `const` item
    static B;
    //~^ ERROR associated `static` items are not allowed
    //~| ERROR missing type for `static` item
}

fn main() {}
