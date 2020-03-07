// Adapted from rust-lang/rust#58813

// revisions: rpass1 cfail2

#[cfg(rpass1)]
pub trait T2 { }
#[cfg(cfail2)]
pub trait T2: T1 { }
//[cfail2]~^ ERROR cycle detected when computing the supertraits of `T2`

pub trait T1: T2 { }

fn main() { }
