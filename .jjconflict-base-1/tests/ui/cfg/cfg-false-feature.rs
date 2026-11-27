// Features above `cfg(false)` are in effect in a fully unconfigured crate (issue #104633).

//@ check-pass
//@ compile-flags: --crate-type lib
//@ edition: 2018

#![feature(decl_macro)]
#![cfg(false)]
#![feature(try_blocks)]

macro mac() {} // OK

trait A = Clone; //~ WARN trait aliases are experimental
                 //~| WARN unstable syntax can change at any point in the future

fn main() {
    try {} //~ WARN `try` blocks are unstable
           //~| WARN unstable syntax can change at any point in the future
}
