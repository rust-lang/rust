#![feature(rustc_attrs)]
#![allow(unused_unconstructable_pub_structs)]

#[cfg_attr(any(rpass2, rpass3), doc = "Some comment")]
pub struct Foo;

pub struct Wrapper(Foo);
