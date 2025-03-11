//@ known-bug: #137888
#![feature(generic_const_exprs)]
macro_rules! empty {
    () => ();
}
fn bar<const N: i32>() -> [(); {
       empty! {};
       N
   }] {
}
fn main() {}
