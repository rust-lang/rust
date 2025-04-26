#![feature(const_trait_impl)]
#![allow(bare_trait_objects)]

//@ check-pass

struct S;
trait T {}

const impl S {}

const impl dyn T {}

fn main() {}
