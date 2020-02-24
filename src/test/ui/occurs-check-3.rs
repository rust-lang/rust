// From Issue #778
#![allow(type_param_on_variant_ctor)]

enum Clam<T> { A(T) }
fn main() { let c; c = Clam::A(c); match c { Clam::A::<isize>(_) => { } } }
//~^ ERROR mismatched types
