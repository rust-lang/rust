#![feature(rustc_attrs)]

#[cfg_attr(any(rpass2, rpass3), doc = "Some comment")]
pub struct Foo;

pub struct Wrapper(Foo);
