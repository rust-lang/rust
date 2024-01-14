#![warn(dereferencing_mut_binding)]
struct Foo {}

pub fn main() {
    let mut tups = vec![(Foo {}, Foo {})];
    // The below desugars to &(ref n, mut m).
    for (n, mut m) in &tups {
        //~^ ERROR cannot move out of a shared reference
        //~| WARN dereferencing `mut`
        //~| WARN this changes meaning in Rust 2024
    }
}
