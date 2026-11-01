//@ edition:2015
//@ reference: lex.keywords.strict.edition2018

#![deny(rust_2018_compatibility)]

// Don't make a suggestion for a raw identifier replacement unless raw
// identifiers are enabled.

fn main() {
    let async = 3; //~ ERROR: is a keyword
    //~^ WARN this is accepted in the current edition
}
