//@ run-pass
//@ compile-flags: -Csymbol-mangling-version=v0
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

pub fn f<T>() {}
pub trait Frob<T> {}
fn main() {
    f::<dyn Frob<str>>();
    f::<dyn for<'a> Frob<str>>();
}
