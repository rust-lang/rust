#![feature(rustc_attrs, core)]
#![crate_type = "rlib"]

#[rustc_auto_trait]
pub trait DefaultedTrait { }

pub struct Something<T> { t: T }
