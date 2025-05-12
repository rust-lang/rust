//@ known-bug: #94846
#![feature(generic_const_exprs)]

struct S<const C:() = {}>() where S<{}>:;

pub fn main() {}
