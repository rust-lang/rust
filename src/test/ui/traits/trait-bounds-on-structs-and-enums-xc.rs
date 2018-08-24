// aux-build:trait_bounds_on_structs_and_enums_xc.rs

extern crate trait_bounds_on_structs_and_enums_xc;

use trait_bounds_on_structs_and_enums_xc::{Bar, Foo, Trait};

fn explode(x: Foo<usize>) {}
//~^ ERROR E0277

fn kaboom(y: Bar<f32>) {}
//~^ ERROR E0277

fn main() {
}
