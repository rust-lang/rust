// error-pattern:thread 'main' panicked at 'attempt to negate with overflow'
// compile-flags: -C debug-assertions

#![allow(overflow)]

fn main() {
    let _x = -std::i8::MIN;
}
