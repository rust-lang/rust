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
    let packed_ref = &Packed(Foo);
    let _used = (|| packed_ref.0.use)();
    //~^ ERROR cannot borrow `packed_ref.0` as mutable, as it is behind a `&` reference
}
