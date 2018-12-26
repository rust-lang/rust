// error-pattern:thread 'main' panicked at 'attempt to negate with overflow'
// compile-flags: -C debug-assertions

#![allow(const_err)]

fn main() {
    let _x = -std::i8::MIN;
}
