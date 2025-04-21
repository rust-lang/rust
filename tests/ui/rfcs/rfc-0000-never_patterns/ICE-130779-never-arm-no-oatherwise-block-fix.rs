//@ run-rustfix
#![feature(never_patterns)]
#![allow(incomplete_features, dead_code, unreachable_patterns)]

enum Void {}

fn foo(v: Void) {
    match v {
        ! | //~ ERROR: a trailing `|` is not allowed in an or-pattern
        if true => {} //~ ERROR: a never pattern is always unreachable
    }
}

fn main() {}
