// Test that we can't use another type in place of !

#![feature(never_type)]
#![deny(warnings)]

fn main() {
    let x: ! = "hello"; //~ ERROR mismatched types
}
