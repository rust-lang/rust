// Regression test for #88472, where a suggestion was issued to
// import an inaccessible struct.

#![warn(unused_imports)]
//~^ NOTE: the lint level is defined here

mod a {
    struct Foo;
}

mod b {
    use crate::a::*;
    //~^ WARNING: unused import
    type Bar = Foo;
    //~^ ERROR: cannot find type `Foo` in this scope [E0412]
    //~| NOTE: not found in this scope
    //~| NOTE: this struct exists but is inaccessible
}

mod c {
    enum Eee {}

    mod d {
        enum Eee {}
    }
}

mod e {
    type Baz = Eee;
    //~^ ERROR: cannot find type `Eee` in this scope [E0412]
    //~| NOTE: not found in this scope
    //~| NOTE: these items exist but are inaccessible
}

fn main() {}
