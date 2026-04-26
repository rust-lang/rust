// Adapted from rust-lang/rust#58813

//@ revisions: rpass1 bfail2

#[cfg(rpass1)]
pub trait T2 {}
#[cfg(bfail2)]
pub trait T2: T1 {}
//[bfail2]~^ ERROR cycle detected when computing the super predicates of `T2`

pub trait T1: T2 {}

fn main() {}
