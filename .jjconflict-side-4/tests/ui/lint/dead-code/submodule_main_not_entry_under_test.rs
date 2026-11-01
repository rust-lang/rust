//@ compile-flags: --test

#![deny(dead_code)]

fn main() {}

mod m {
    pub fn main() {} //~ ERROR: function `main` is never used
}
