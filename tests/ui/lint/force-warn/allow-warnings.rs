// --force-warn $LINT causes $LINT (which is warn-by-default) to warn
// despite allowing all warnings in module
//@ compile-flags: --force-warn dead_code
//@ check-pass

#![allow(warnings)]

fn dead_function() {}
//~^ WARN function `dead_function` is never used

fn main() {}
