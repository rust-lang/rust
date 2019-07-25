// Unknown attributes fall back to feature gated custom attributes.

#![feature(custom_inner_attributes)]

#![mutable_doc]
//~^ ERROR cannot find attribute macro `mutable_doc` in this scope

#[dance] mod a {}
//~^ ERROR cannot find attribute macro `dance` in this scope

#[dance] fn main() {}
//~^ ERROR cannot find attribute macro `dance` in this scope
