// --force-warn $LINT causes $LINT (which is deny-by-default) to warn
// despite $LINT being allowed on command line
// compile-flags: -A const_err --force-warn const_err
// check-pass

const C: i32 = 1 / 0;
//~^ WARN any use of this value will cause an error
//~| WARN this was previously accepted by the compiler

fn main() {}
