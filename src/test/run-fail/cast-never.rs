// Test that we can explicitly cast ! to another type

// error-pattern:explicit

#![feature(never_type)]

fn main() {
    let x: ! = panic!();
    let y: u32 = x as u32;
}
