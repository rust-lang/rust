// Validation makes this fail in the wrong place
// Make sure we find these even with many checks disabled.
//@compile-flags: -Zmiri-disable-alignment-check -Zmiri-disable-stacked-borrows -Zmiri-disable-validation

#![allow(unnecessary_transmutes)]
fn main() {
    let b = unsafe { std::mem::transmute::<u8, bool>(2) };
    let _x = b == std::hint::black_box(true); //~ ERROR: interpreting an invalid 8-bit value as a bool
}
