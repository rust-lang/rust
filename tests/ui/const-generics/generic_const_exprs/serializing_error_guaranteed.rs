//@ check-pass

// regression test for #136894.
// I (BoxyUwU) don't know what the underlying cause was here

#![feature(generic_const_exprs)]
#![crate_type = "lib"]
#![allow(incomplete_features, dead_code)]

struct X<T>([(); f::<T>()])
where
    [(); f::<T>()]:;

const fn f<T>() -> usize {
    panic!()
}
