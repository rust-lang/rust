#![feature(const_trait_impl)]
#![allow(bare_trait_objects)]

//@ check-pass

struct S;
trait T {}

impl const S {}

impl const dyn T {}

fn main() {}
