// error-pattern:thread 'main' panicked at 'attempt to shift right with overflow'
// compile-flags: -C debug-assertions

#![warn(arithmetic_overflow)]
#![warn(const_err)]
#![feature(const_indexing)]

fn main() {
    let _n = 1i64 >> [64][0];
}
