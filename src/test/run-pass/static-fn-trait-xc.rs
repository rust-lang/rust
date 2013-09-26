// aux-build:static_fn_trait_xc_aux.rs
// xfail-fast

extern mod mycore(name ="static_fn_trait_xc_aux");

use mycore::num;

pub fn main() {
    let _1: f64 = num::Num2::from_int2(1i);
}
