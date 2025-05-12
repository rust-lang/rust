//@ known-bug: #136894
#![feature(generic_const_exprs)]
#![crate_type = "lib"]
#![allow(incomplete_features, dead_code)]

struct X<T>([(); f::<T>()]) where [(); f::<T>()]:;

const fn f<T>() -> usize { panic!() }
