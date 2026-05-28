//@ check-pass
// Regression test for issue #61651
// Verifies that we don't try to constrain inference
// variables due to the presence of multiple applicable
// marker trait impls

#![feature(marker_trait_attr)]

#[marker] // Remove this line and it works?!?
trait Foo<T> {}
impl Foo<u16> for u8 {}
impl Foo<[u8; 1]> for u8 {}
fn foo<T: Foo<U>, U>(_: T) -> U { unimplemented!() }

fn main() {
    let _: u16 = foo(0_u8);
}
