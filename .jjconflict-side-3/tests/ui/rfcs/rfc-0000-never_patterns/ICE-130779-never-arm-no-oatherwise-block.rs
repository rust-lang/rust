#![feature(never_patterns)]
#![allow(incomplete_features)]

enum E { A }

fn main() {
    match E::A {
        ! | //~ ERROR: a trailing `|` is not allowed in an or-pattern
        //~^ ERROR: mismatched types
        if true => {} //~ ERROR: a never pattern is always unreachable
    }
}
