#![feature(rustc_attrs)]

#[rustc_coherence_future_impls]
pub trait Unrestricted {} //~ ERROR requires an impl-restricted trait

impl Unrestricted for () {}

fn main() {}
