// aux-build:trait_default_method_xc_aux.rs

extern crate trait_default_method_xc_aux as aux;
use aux::A;

pub struct a_struct { pub x: isize }

impl A for a_struct {
    fn f(&self) -> isize { 10 }
}

// This function will need to get inlined, and badness may result.
pub fn welp<A>(x: A) -> A {
    let a = a_struct { x: 0 };
    a.g();
    x
}
