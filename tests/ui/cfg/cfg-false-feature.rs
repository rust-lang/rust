// Features above `cfg(FALSE)` are in effect in a fully unconfigured crate (issue #104633).

// check-pass
// compile-flags: --crate-type lib

#![feature(decl_macro)]
#![cfg(FALSE)]
#![feature(box_syntax)]

macro mac() {} // OK

trait A = Clone; //~ WARN trait aliases are experimental
                 //~| WARN unstable syntax can change at any point in the future

fn main() {
    let box _ = Box::new(0); //~ WARN box pattern syntax is experimental
                             //~| WARN unstable syntax can change at any point in the future
}
