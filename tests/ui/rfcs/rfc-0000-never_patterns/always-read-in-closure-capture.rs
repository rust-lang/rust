//@ check-pass

// Make sure that the closure captures `s` so it can perform a read of `s`.

#![feature(never_patterns)]
#![allow(incomplete_features)]

enum Void {}

fn by_value(s: Void) {
    move || {
        let ! = s;
    };
}

fn main() {}
