#![crate_type = "lib"]
#![feature(negative_impls)]

pub trait Error {}
impl !Error for &str {}
