#![crate_type = "rlib"]

// Suggestions for range patterns should not perform span manipulations that
// assume the range token is ASCII, because it could have been recovered from
// similar-looking Unicode characters.
//
// Regression test for <https://github.com/rust-lang/rust/issues/155799>.

// These dots are U+00B7 MIDDLE DOT, not an ASCII period.
fn dot_dot_dot() { ··· }
//~^ ERROR unknown start of token
//~| ERROR unknown start of token
//~| ERROR unknown start of token
//~| ERROR unexpected token: `...`
//~| ERROR inclusive range with no end

fn dot_dot_dot_eq() { ···= }
//~^ ERROR unknown start of token
//~| ERROR unknown start of token
//~| ERROR unknown start of token
//~| ERROR unexpected token: `...`
//~| ERROR unexpected `=` after inclusive range

fn dot_dot_dot_gt() { ···> }
//~^ ERROR unknown start of token
//~| ERROR unknown start of token
//~| ERROR unknown start of token
//~| ERROR unexpected token: `...`
//~| ERROR inclusive range with no end
//~| ERROR expected one of `;` or `}`, found `>`

fn dot_dot_eq() { ··= }
//~^ ERROR unknown start of token
//~| ERROR unknown start of token
//~| ERROR inclusive range with no end

fn dot_dot_eq_eq() { ··== }
//~^ ERROR unknown start of token
//~| ERROR unknown start of token
//~| ERROR unexpected `=` after inclusive range

fn dot_dot_eq_gt() { ··=> }
//~^ ERROR unknown start of token
//~| ERROR unknown start of token
//~| ERROR unexpected `>` after inclusive range
//~| ERROR expected one of `;` or `}`, found `>`
