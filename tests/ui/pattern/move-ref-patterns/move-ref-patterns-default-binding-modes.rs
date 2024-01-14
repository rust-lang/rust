#![warn(dereferencing_mut_binding)]
fn main() {
    struct U;

    // A tuple is a "non-reference pattern".
    // A `mut` binding pattern resets the binding mode to by-value.

    let p = (U, U);
    let (a, mut b) = &p;
    //~^ ERROR cannot move out of a shared reference
    //~| WARN dereferencing `mut`
    //~| WARN this changes meaning in Rust 2024
}
