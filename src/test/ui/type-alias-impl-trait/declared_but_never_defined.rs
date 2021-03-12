// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

fn main() {}

// declared but never defined
type Bar = impl std::fmt::Debug; //~ ERROR could not find defining uses
