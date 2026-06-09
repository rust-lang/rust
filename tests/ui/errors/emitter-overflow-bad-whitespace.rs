// Invalid whitespace (not listed here: https://doc.rust-lang.org/reference/whitespace.html
// e.g. \u{a0}) before any other syntax on the line should not cause any integer overflow
// in the emitter, even when the terminal width causes the line to be truncated.
//
// issue #132918

//@ check-fail
//@ needs-rustc-debug-assertions
//@ compile-flags: --diagnostic-width=1
                                         fn main() {              return;              }
//~^ ERROR unknown start of token: \u{a0}
