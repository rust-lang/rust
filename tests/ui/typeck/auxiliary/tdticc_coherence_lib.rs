#![feature(auto_traits, core)]
#![crate_type = "rlib"]
#![allow(unused_unconstructable_pub_structs)]

pub auto trait DefaultedTrait { }

pub struct Something<T> { t: T }
