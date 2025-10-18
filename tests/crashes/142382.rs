//@ known-bug: rust-lang/rust#142382

#![feature(generic_const_parameter_types)]
#![feature(adt_const_params)]
#![feature(unsized_const_params)]

struct Bar<'a, const N: &'a u32>;
fn foo(&self) -> Bar<0> { todo!(); }

pub fn main() {}
