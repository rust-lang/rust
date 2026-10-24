enum Foo {
    Foo_(Bar)
}

enum Bar {
    //~^ ERROR recursive type `Bar` has infinite size
    BarNone,
    BarSome(Bar)
}

fn main() {
}
