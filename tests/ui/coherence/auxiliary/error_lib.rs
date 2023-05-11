#![crate_type = "lib"]
#![feature(negative_impls)]
#![feature(with_negative_coherence)]

pub trait Error {}
impl !Error for &str {}
