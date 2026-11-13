#![feature(precise_capturing)]

pub fn cross_crate_empty() -> impl Sized + use<> {}

pub fn cross_crate_missing() -> impl Sized {}

pub fn cross_crate_args<'a, T, const N: usize>() -> impl Sized + use<'a, T, N> {}
