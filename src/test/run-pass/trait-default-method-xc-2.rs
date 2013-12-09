// xfail-fast
// aux-build:trait_default_method_xc_aux.rs
// aux-build:trait_default_method_xc_aux_2.rs


extern mod aux = "trait_default_method_xc_aux";
extern mod aux2 = "trait_default_method_xc_aux_2";
use aux::A;
use aux2::{a_struct, welp};


fn main () {

    let a = a_struct { x: 0 };
    let b = a_struct { x: 1 };

    assert_eq!(0i.g(), 10);
    assert_eq!(a.g(), 10);
    assert_eq!(a.h(), 11);
    assert_eq!(b.g(), 10);
    assert_eq!(b.h(), 11);
    assert_eq!(A::lurr(&a, &b), 21);

    welp(&0);
}
