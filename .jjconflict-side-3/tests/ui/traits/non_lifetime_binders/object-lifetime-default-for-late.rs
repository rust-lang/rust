//@ check-pass
//@ compile-flags: --crate-type=lib

#![feature(non_lifetime_binders)]

pub fn f<T>() where for<U> (T, U): Copy {}
