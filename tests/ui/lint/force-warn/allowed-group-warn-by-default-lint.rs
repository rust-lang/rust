// --force-warn $LINT causes $LINT (which is warn-by-default) to warn
// despite $LINT_GROUP (which contains $LINT) being allowed
//@ compile-flags: --force-warn bare_trait_objects
//@ check-pass

#![allow(rust_2018_idioms)]

pub trait SomeTrait {}

pub fn function(_x: Box<SomeTrait>) {}
//~^ WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition

fn main() {}
