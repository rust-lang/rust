// --force-warn $LINT casuses $LINT to warn despite --cap-lints
// set to allow
// compile-flags: --cap-lints allow  --force-warn bare_trait_objects
// check-pass

pub trait SomeTrait {}

pub fn function(_x: Box<SomeTrait>) {}
//~^ WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition
//~| WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition
//~| WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition

fn main() {}
