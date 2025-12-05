//@ check-pass

#![deny(warnings)]

//! [usize::Item]

pub trait Foo { type Item; }
pub trait Bar { type Item; }

impl Foo for usize { type Item = u32; }
impl Bar for usize { type Item = i32; }
