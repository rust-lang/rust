// check-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

pub trait Foo<const B: bool> {}
pub fn bar<T: Foo<{ true }>>() {}

fn main() {}
