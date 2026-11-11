//@ check-pass

#![feature(import_trait_associated_functions)]

trait Trait: Default {
    fn f() -> Self { Default::default() }
    fn g() -> Self { Default::default() }
}

impl Trait for u8 {}

use Trait::*;

fn main() {
    let _: u8 = f();
    let _: u8 = g();
}
