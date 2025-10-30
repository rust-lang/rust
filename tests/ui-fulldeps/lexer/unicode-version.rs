// This test is used to validate which version of Unicode is used for parsing
// identifiers. If the Unicode version changes, it should also be updated in
// the reference at
// https://github.com/rust-lang/reference/blob/HEAD/src/identifiers.md.

//@ run-pass
//@ check-run-results
//@ ignore-cross-compile
//@ reference: ident.unicode
//@ reference: ident.normalization

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_lexer;
extern crate rustc_parse;

fn main() {
    println!("Checking if Unicode version changed.");
    println!(
        "If the Unicode version changes are intentional, \
         it should also be updated in the reference at \
         https://github.com/rust-lang/reference/blob/HEAD/src/identifiers.md."
    );
    println!("Unicode version of unicode-ident is: {:?}", rustc_lexer::UNICODE_IDENT_VERSION);
    println!(
        "Unicode version of unicode-normalization is: {:?}",
        rustc_parse::UNICODE_NORMALIZATION_VERSION
    );
}
