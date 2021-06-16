// compile-flags: --force-warns bare_trait_objects -Zunstable-options
// check-pass

#![allow(rust_2018_idioms)]

pub trait SomeTrait {}

pub fn function(_x: Box<SomeTrait>) {}
//~^ WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition

fn main() {}
