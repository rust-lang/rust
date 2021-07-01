// compile-flags: --cap-lints allow  --force-warns bare_trait_objects -Zunstable-options
// check-pass

pub trait SomeTrait {}

pub fn function(_x: Box<SomeTrait>) {}
//~^ WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition

fn main() {}
