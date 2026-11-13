/// Regression test for <https://github.com/rust-lang/rust/issues/155836>.
///
/// SsaRangeProp pass used to fail the assert when encountering self-dominating block
/// e.g. small loops like in `a`

//@ compile-flags: -Copt-level=2
//@ build-pass

use std::hint::black_box;
fn a(d: u8) {
    loop {
        1 % d;
    }
}
pub fn e(d: u8) {
    if d == 0 || black_box(false) {
        a(d);
    }
}
fn main() {
    e(1);
}
