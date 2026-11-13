//@ run-pass
//@ edition: 2021

#![feature(ergonomic_clones)]
#![expect(incomplete_features)]

use std::clone::UseCloned;


#[derive(Clone)]
#[repr(align(32))]
struct Foo;

impl UseCloned for Foo {}

#[repr(packed)]
struct Packed(Foo);

fn main() {
    let packed = Packed(Foo);
    let _used = (|| packed.0.use)();

    let mut packed = packed;
    let packed_mut = &mut packed;
    let _used = (|| packed_mut.0.use)();
}
