//@ run-pass
//@ run-rustfix
//@ rustfix-only-machine-applicable
#![allow(unused_variables, unused_mut)]
#![warn(dereferencing_mut_binding)]
fn main() {
    // A tuple is a "non-reference pattern".
    // A `mut` binding pattern resets the binding mode to by-value
    // in edition <= 2021.

    let mut p = (0u8, 0u8);
    let (a, mut b) = &mut p;
    //~^ WARN dereferencing `mut`
    //~| WARN this changes meaning in Rust 2024
    let _: u8 = b;

    let (a, mut b) = &p;
    //~^ WARN dereferencing `mut`
    //~| WARN this changes meaning in Rust 2024
    let _: u8 = b;

    let (a, mut b @ _) = &mut p;
    //~^ WARN dereferencing `mut`
    //~| WARN this changes meaning in Rust 2024
    let _: u8 = b;

    let (a, mut b @ _) = &p;
    //~^ WARN dereferencing `mut`
    //~| WARN this changes meaning in Rust 2024
    let _: u8 = b;

    let mut p = (&0u8, &0u8);
    let (a, mut b) = &mut p;
    //~^ WARN dereferencing `mut`
    //~| WARN this changes meaning in Rust 2024
    let _: &u8 = b;

    let (a, mut b) = &p;
    //~^ WARN dereferencing `mut`
    //~| WARN this changes meaning in Rust 2024
    let _: &u8 = b;

    let (a, mut b @ _) = &mut p;
    //~^ WARN dereferencing `mut`
    //~| WARN this changes meaning in Rust 2024
    let _: &u8 = b;

    let (a, mut b @ _) = &p;
    //~^ WARN dereferencing `mut`
    //~| WARN this changes meaning in Rust 2024
    let _: &u8 = b;
}
