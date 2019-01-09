// aux-build:trait_bounds_on_structs_and_enums_xc.rs

extern crate trait_bounds_on_structs_and_enums_xc;

use trait_bounds_on_structs_and_enums_xc::{Bar, Foo, Trait};

fn main() {
    let foo = Foo {
    //~^ ERROR E0277
        x: 3
    };
    let bar: Bar<f64> = return;
    //~^ ERROR E0277
    let _ = bar;
}
