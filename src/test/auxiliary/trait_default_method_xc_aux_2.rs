// aux-build:trait_default_method_xc_aux.rs

extern mod aux(name = "trait_default_method_xc_aux");
use aux::A;

pub struct a_struct { x: int }

impl A for a_struct {
    fn f(&self) -> int { 10 }
}

// This function will need to get inlined, and badness may result.
pub fn welp<A>(x: A) -> A {
    let a = a_struct { x: 0 };
    a.g();
    x
}
