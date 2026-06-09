// --force-warn $LINT causes $LINT (which is warn-by-default) to warn
// despite $LINT being allowed in module
//@ compile-flags: --force-warn dead_code
//@ check-pass

#![allow(dead_code)]

fn dead_function() {}
//~^ WARN function `dead_function` is never used

fn main() {}
