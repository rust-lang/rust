// Test that we can't use another type in place of !

#![deny(warnings)]

fn main() {
    let x: ! = "hello"; //~ ERROR mismatched types
}
