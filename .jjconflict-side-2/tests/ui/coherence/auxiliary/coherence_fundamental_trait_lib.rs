#![crate_type = "rlib"]
#![feature(fundamental)]

pub trait Misc {}

#[fundamental]
pub trait Fundamental<T> {}
