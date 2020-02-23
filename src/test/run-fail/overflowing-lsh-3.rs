// error-pattern:thread 'main' panicked at 'attempt to shift left with overflow'
// compile-flags: -C debug-assertions

#![warn(arithmetic_overflow)]
#![warn(const_err)]

fn main() {
    let _x = 1_u64 << 64;
}
