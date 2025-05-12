//@ known-bug: #127643

#![feature(generic_const_items, associated_const_equality)]
#![expect(incomplete_features)]

trait Foo {
    const ASSOC<const N: u32>: u32;
}

impl Foo for () {
    const ASSOC<const N: u32>: u32 = N;
}

fn bar<const N: u64, T: Foo<ASSOC<N> = { N }>>() {}

fn main() {
    bar::<10_u64, ()>();
}
