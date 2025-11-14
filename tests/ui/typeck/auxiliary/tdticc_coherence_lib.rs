#![feature(auto_traits, core)]
#![crate_type = "rlib"]

#![allow(unconstructable_pub_struct)]

pub auto trait DefaultedTrait { }

pub struct Something<T> { t: T }
