// error-pattern:thread 'main' panicked at 'attempt to add with overflow'
// compile-flags: -C debug-assertions

#![allow(overflow)]

fn main() {
    let _x = 200u8 + 200u8 + 200u8;
}
