// check that error types in coherence do not cause error cascades.

// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

trait Foo {}

impl Foo for i8 {}
impl Foo for i16 {}
impl Foo for i32 {}
impl Foo for i64 {}
impl Foo for DoesNotExist {}
//[old]~^ ERROR cannot find type `DoesNotExist` in this scope
//[re]~^^ ERROR E0412
impl Foo for u8 {}
impl Foo for u16 {}
impl Foo for u32 {}
impl Foo for u64 {}

fn main() {}
