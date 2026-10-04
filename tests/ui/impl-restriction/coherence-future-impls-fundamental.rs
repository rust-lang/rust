#![feature(fundamental, rustc_attrs)]

#[fundamental]
#[rustc_coherence_future_impls]
pub trait Invalid {} //~ ERROR cannot be used with `#[fundamental]`

impl Invalid for () {}

fn main() {}
