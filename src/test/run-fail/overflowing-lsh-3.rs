// error-pattern:thread 'main' panicked at 'attempt to shift left with overflow'
// compile-flags: -C debug-assertions

#![warn(exceeding_bitshifts)]

fn main() {
    let _x = 1_u64 << 64;
}
