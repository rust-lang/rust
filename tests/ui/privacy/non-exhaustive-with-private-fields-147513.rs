//@ aux-build:non_exhaustive_with_private.rs

extern crate non_exhaustive_with_private;

use non_exhaustive_with_private::{Bar, Foo};

fn main() {
    let foo = Foo {
        //~^ ERROR cannot create non-exhaustive struct using struct expression
        my_field: 10,
    };

    let bar = Bar {
        //~^ ERROR cannot create non-exhaustive struct using struct expression
        my_field: 10,
    };
}
