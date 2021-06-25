// compile-flags: --force-warns rust-2018-idioms -Zunstable-options
// check-pass

#![allow(bare_trait_objects)]

pub trait SomeTrait {}

pub fn function(_x: Box<SomeTrait>) {}
//~^ WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition

fn main() {}
