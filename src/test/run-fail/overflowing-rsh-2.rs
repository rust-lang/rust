// error-pattern:thread 'main' panicked at 'attempt to shift right with overflow'
// compile-flags: -C debug-assertions

#![warn(exceeding_bitshifts)]
#![warn(const_err)]

fn main() {
    let _x = -1_i32 >> -1;
}
