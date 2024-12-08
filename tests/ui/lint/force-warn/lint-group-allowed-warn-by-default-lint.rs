// --force-warn $LINT_GROUP causes $LINT (which is warn-by-default) to warn
// despite $LINT being allowed in module
//@ compile-flags: --force-warn rust-2018-idioms
//@ check-pass

#![allow(bare_trait_objects)]

pub trait SomeTrait {}

pub fn function(_x: Box<SomeTrait>) {}
//~^ WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition

fn main() {}
