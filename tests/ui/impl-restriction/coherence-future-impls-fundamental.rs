#![feature(fundamental, rustc_attrs)]

#[fundamental]
#[rustc_coherence_future_impls]
pub trait Invalid {}
//~^ ERROR `#[rustc_coherence_future_impls]` cannot be used with `#[fundamental]`

fn main() {}
