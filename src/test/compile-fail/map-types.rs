use std;
use std::map;
use std::map::HashMap;
use std::map::Map;

// Test that trait types printed in error msgs include the type arguments.

fn main() {
    let x: Map<~str,~str> = map::str_hash::<~str>() as Map::<~str,~str>;
    let y: Map<uint,~str> = x;
    //~^ ERROR mismatched types: expected `@std::map::Map<uint,~str>`
}
