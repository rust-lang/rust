// Regression test for issue #90870.

#![allow(dead_code)]

const fn f(a: &u8, b: &u8) -> bool {
    //~^ HELP: add `#![feature(const_cmp)]` to the crate attributes to enable
    //~| HELP: add `#![feature(const_cmp)]` to the crate attributes to enable
    //~| HELP: add `#![feature(const_cmp)]` to the crate attributes to enable
    a == b
    //~^ ERROR: cannot call conditionally-const operator in constant functions
    //~| ERROR: `PartialEq` is not yet stable as a const trait
    //~| HELP: consider dereferencing here
    //~| HELP: add `#![feature(const_trait_impl)]` to the crate attributes to enable
}

const fn g(a: &&&&i64, b: &&&&i64) -> bool {
    a == b
    //~^ ERROR: cannot call conditionally-const operator in constant functions
    //~| ERROR: `PartialEq` is not yet stable as a const trait
    //~| HELP: consider dereferencing here
    //~| HELP: add `#![feature(const_trait_impl)]` to the crate attributes to enable
}

const fn h(mut a: &[u8], mut b: &[u8]) -> bool {
    while let ([l, at @ ..], [r, bt @ ..]) = (a, b) {
        if l == r {
        //~^ ERROR: cannot call conditionally-const operator in constant functions
        //~| ERROR: `PartialEq` is not yet stable as a const trait
        //~| HELP: consider dereferencing here
        //~| HELP: add `#![feature(const_trait_impl)]` to the crate attributes to enable
            a = at;
            b = bt;
        } else {
            return false;
        }
    }

    a.is_empty() && b.is_empty()
}

fn main() {}
