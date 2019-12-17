// error-pattern:thread 'main' panicked at 'attempt to negate with overflow'
// compile-flags: -C debug-assertions

#![allow(const_err)]

fn main() {
    let _x = -i8::MIN;
}
