use std;
import std::map;
import std::map::map;

// Test that iface types printed in error msgs include the type arguments.

fn main() {
    let x: map<uint,str> = map::new_str_hash::<str>();
    //!^ ERROR mismatched types: expected `std::map::map<uint,str>`
}