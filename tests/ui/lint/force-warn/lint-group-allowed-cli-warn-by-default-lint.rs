// --force-warn $LINT_GROUP causes $LINT (which is warn-by-default) to warn
// despite $LINT being allowed on command line
// compile-flags: -A bare-trait-objects --force-warn rust-2018-idioms
// check-pass

pub trait SomeTrait {}

pub fn function(_x: Box<SomeTrait>) {}
//~^ WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition

fn main() {}
