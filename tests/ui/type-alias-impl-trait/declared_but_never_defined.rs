#![feature(type_alias_impl_trait)]

fn main() {}

// declared but never defined
type Bar = impl std::fmt::Debug; //~ ERROR unconstrained opaque type
