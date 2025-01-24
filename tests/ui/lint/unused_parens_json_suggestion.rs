//@ compile-flags: --error-format json
//@ error-pattern:unnecessary parentheses
//@ run-rustfix

// The output for humans should just highlight the whole span without showing
// the suggested replacement, but we also want to test that suggested
// replacement only removes one set of parentheses, rather than naïvely
// stripping away any starting or ending parenthesis characters—hence this
// test of the JSON error format.

#![deny(unused_parens)]
#![allow(unreachable_code, unused_braces)]

fn main() {
    // We want to suggest the properly-balanced expression `1 / (2 + 3)`, not
    // the malformed `1 / (2 + 3`
    let _a = (1 / (2 + 3));
    f();
}

fn f() -> bool {
    loop {
        if (break { return true }) {
        }
    }
    false
}
