// Unknown attributes fall back to unstable custom attributes.

#![feature(custom_inner_attributes)]

#![mutable_doc]
//~^ ERROR cannot find attribute `mutable_doc`

#[dance] mod a {}
//~^ ERROR cannot find attribute `dance`

#[dance] fn main() {}
//~^ ERROR cannot find attribute `dance`
