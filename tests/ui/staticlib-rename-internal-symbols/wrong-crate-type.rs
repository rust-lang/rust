//@ compile-flags: -Zstaticlib-rename-internal-symbols --crate-type bin

#![feature(no_core)]
#![no_core]
#![no_main]

//~? ERROR can only be used with `--crate-type staticlib`
