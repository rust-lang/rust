// --force-warn $LINT_GROUP causes $LINT to warn despite $LINT being
// allowed in module and cap-lints set to warn
//@ compile-flags: --cap-lints warn  --force-warn rust-2021-compatibility
//@ check-pass
#![allow(ellipsis_inclusive_range_patterns)]

pub fn f() -> bool {
    let x = 123;
    match x {
        0...100 => true,
        //~^ WARN range patterns are deprecated
        //~| WARN this is accepted in the current edition
        _ => false,
    }
}

fn main() {}
