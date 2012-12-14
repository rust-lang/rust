// aux-build:static_fn_trait_xc_aux.rs
// xfail-fast

extern mod mycore(name ="static_fn_trait_xc_aux");

use mycore::num;

fn main() {
    let _1:float = num::from_int2(1i);
}
