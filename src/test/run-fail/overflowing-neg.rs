// error-pattern:thread 'main' panicked at 'attempt to negate with overflow'
// compile-flags: -C debug-assertions

#![allow(arithmetic_overflow)]

fn main() {
    let _x = -std::i8::MIN;
}
