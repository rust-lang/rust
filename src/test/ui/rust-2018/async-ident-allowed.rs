// edition:2015

#![deny(rust_2018_compatibility)]

// Don't make a suggestion for a raw identifier replacement unless raw
// identifiers are enabled.

fn main() {
    let async = 3; //~ ERROR: is a keyword
    //~^ WARN previously accepted
}
