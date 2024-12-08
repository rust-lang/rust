// Regression test for #88472, where a suggestion was issued to
// import an inaccessible struct.

#![warn(unused_imports)]
//~^ NOTE: the lint level is defined here

mod a {
    struct Foo;
    //~^ NOTE: struct `a::Foo` exists but is inaccessible
    //~| NOTE: not accessible
}

mod b {
    use crate::a::*;
    //~^ WARNING: unused import
    type Bar = Foo;
    //~^ ERROR: cannot find type `Foo` in this scope [E0412]
    //~| NOTE: not found in this scope
}

mod c {
    enum Eee {}
    //~^ NOTE: these enums exist but are inaccessible
    //~| NOTE: `c::Eee`: not accessible

    mod d {
        enum Eee {}
        //~^ NOTE: `c::d::Eee`: not accessible
    }
}

mod e {
    type Baz = Eee;
    //~^ ERROR: cannot find type `Eee` in this scope [E0412]
    //~| NOTE: not found in this scope
}

fn main() {}
