// Suggestions for `...` in patterns should not perform span manipulations that
// assume the token is ASCII, because it could have been recovered from
// similar-looking Unicode characters.
//
// Regression test for https://github.com/rust-lang/rust/issues/156316.

// These dots are U+00B7 MIDDLE DOT, not ASCII periods.
impl S { fn f(···>) }
//~^ ERROR unknown start of token
//~| ERROR unknown start of token
//~| ERROR unknown start of token
//~| ERROR unexpected `...`
//~| ERROR expected `:`, found `>`
//~| ERROR expected one of
//~| ERROR associated function in `impl` without body
//~| ERROR cannot find type `S` in this scope
//~| ERROR missing pattern for `...` argument
//~| WARN this was previously accepted by the compiler

fn main() {}
