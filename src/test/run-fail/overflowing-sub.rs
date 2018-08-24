// error-pattern:thread 'main' panicked at 'attempt to subtract with overflow'
// compile-flags: -C debug-assertions

#![allow(const_err)]

fn main() {
    let _x = 42u8 - (42u8 + 1);
}
