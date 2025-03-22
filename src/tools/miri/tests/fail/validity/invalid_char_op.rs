// Validation makes this fail in the wrong place
// Make sure we find these even with many checks disabled.
//@compile-flags: -Zmiri-disable-alignment-check -Zmiri-disable-stacked-borrows -Zmiri-disable-validation

#![allow(unnecessary_transmutes)]
fn main() {
    let c = 0xFFFFFFu32;
    assert!(std::char::from_u32(c).is_none());
    let c = unsafe { std::mem::transmute::<u32, char>(c) };
    let _x = c == 'x'; //~ ERROR: interpreting an invalid 32-bit value as a char
}
