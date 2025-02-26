//@ run-pass
//@ aux-build:static_fn_inline_xc_aux.rs


extern crate static_fn_inline_xc_aux as mycore;

use mycore::num;

pub fn main() {
    let _1: f64 = num::Num2::from_int2(1);
}
