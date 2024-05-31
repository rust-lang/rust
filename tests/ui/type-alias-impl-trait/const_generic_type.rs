//@edition: 2021

#![feature(type_alias_impl_trait)]
type Bar = impl std::fmt::Display;

async fn test<const N: crate::Bar>() {}
//~^ ERROR: type annotations needed
//~| ERROR: `Bar` is forbidden as the type of a const generic parameter

fn main() {}
