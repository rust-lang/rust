//@ check-pass
//@ compile-flags: -Zstaticlib-hide-internal-symbols --crate-type bin

#![feature(no_core)]
#![no_core]
#![no_main]

//~? WARN has no effect without `--crate-type staticlib`
