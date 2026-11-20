//! Ensure that "forbid" always trumps" allow" in CLI arguments, no matter the order
//! and no matter whether it is used with a lint group vs an individual lint.
// ignore-tidy-linelength
//@ revisions: forbid-first-group allow-first-group forbid-first-lint allow-first-lint forbid-first-mix1 allow-first-mix1 forbid-first-mix2 allow-first-mix2
//@[forbid-first-group] compile-flags: -F unused -A unused
//@[allow-first-group] compile-flags: -A unused -F unused
//@[forbid-first-lint] compile-flags: -F unused_variables -A unused_variables
//@[allow-first-lint] compile-flags: -A unused_variables -F unused_variables
//@[forbid-first-mix1] compile-flags: -F unused -A unused_variables
//@[allow-first-mix1] compile-flags: -A unused_variables -F unused
//@[forbid-first-mix2] compile-flags: -F unused_variables -A unused
//@[allow-first-mix2] compile-flags: -A unused -F unused_variables

fn main() {
    let x = 1;
    //~^ ERROR unused variable: `x`
}
