// Unknown attributes fall back to unstable custom attributes.

#![feature(custom_inner_attributes)]

#![mutable_doc]
//~^ ERROR cannot find attribute `mutable_doc` in this scope

#[dance] mod a {}
//~^ ERROR cannot find attribute `dance` in this scope

#[dance] fn main() {}
//~^ ERROR cannot find attribute `dance` in this scope
