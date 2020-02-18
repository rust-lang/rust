// error-pattern:thread 'main' panicked at 'attempt to subtract with overflow'
// compile-flags: -C debug-assertions

#![allow(arithmetic_overflow)]

fn main() {
    let _x = 42u8 - (42u8 + 1);
}
