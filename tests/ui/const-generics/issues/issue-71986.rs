//@ check-pass

pub trait Foo<const B: bool> {}
pub fn bar<T: Foo<{ true }>>() {}

fn main() {}
