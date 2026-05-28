#![feature(auto_traits, core)]
#![crate_type = "rlib"]

pub auto trait DefaultedTrait { }

pub struct Something<T> { t: T }
