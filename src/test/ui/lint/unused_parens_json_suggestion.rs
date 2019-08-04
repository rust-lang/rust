// compile-flags: --error-format -Zunstable-options
// build-pass (FIXME(62277): could be check-pass?)
// run-rustfix

// The output for humans should just highlight the whole span without showing
// the suggested replacement, but we also want to test that suggested
// replacement only removes one set of parentheses, rather than naïvely
// stripping away any starting or ending parenthesis characters—hence this
// test of the JSON error format.

#![allow(unreachable_code)]
#![deny(unused_parens)]

fn main() {
    // We want to suggest the properly-balanced expression `1 / (2 + 3)`, not
    // the malformed `1 / (2 + 3`
    let _a = (1 / (2 + 3)); //~ ERROR: unused parentheses wrap expression
    f();
}

fn f() -> bool {
    loop {
        if (break { return true }) {
        }
    }
    false
}
