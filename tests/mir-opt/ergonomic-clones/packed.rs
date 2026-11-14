//@ skip-filecheck

#![feature(ergonomic_clones)]
#![expect(incomplete_features)]

use std::clone::UseCloned;

#[allow(dead_code)]
#[derive(Clone)]
struct Foo(u128);

impl UseCloned for Foo {}

#[repr(packed)]
struct Packed(Foo);

// EMIT_MIR packed.packed_use.built.after.mir
pub fn packed_use(packed: Packed) -> Foo {
    packed.0.use
}

// EMIT_MIR packed.packed_ref_use.built.after.mir
pub fn packed_ref_use(packed: &mut Packed) -> Foo {
    packed.0.use
}
