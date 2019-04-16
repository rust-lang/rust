// ignore-cloudabi
// ignore-windows
// compile-flags: --error-format pretty-json -Zunstable-options --json-rendered=termcolor

// The output for humans should just highlight the whole span without showing
// the suggested replacement, but we also want to test that suggested
// replacement only removes one set of parentheses, rather than naïvely
// stripping away any starting or ending parenthesis characters—hence this
// test of the JSON error format.

fn main() {
    let x: Iter;
}
