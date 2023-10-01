#![feature(type_alias_impl_trait)]

pub type Tait = impl Iterator<Item = (&'db Key, impl Iterator)>;
//~^ ERROR use of undeclared lifetime name `'db`
//~| ERROR cannot find type `Key` in this scope
//~| ERROR unconstrained opaque type
//~| ERROR unconstrained opaque type

pub fn main() {}
