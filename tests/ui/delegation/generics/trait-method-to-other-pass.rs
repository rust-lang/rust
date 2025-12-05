//@ run-pass

#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod to_reuse {
    pub fn foo<T>(x: T) -> T { x }
}

trait Trait1<T, U> {
    fn foo(&self, _: T, x: U) -> U { x }
}

#[derive(Default)]
struct F;

impl<T, U> Trait1<T, U> for F {}

trait Trait2<T> {
    fn get_f(&self) -> &F { &F }
    reuse Trait1::foo as bar { self.get_f() }
    reuse to_reuse::foo as baz;
}

impl Trait2<u64> for F {}

fn main() {
    assert_eq!(F.bar(1u8, 2u16), 2u16);
    assert_eq!(F::baz(1u8), 1u8);
}
