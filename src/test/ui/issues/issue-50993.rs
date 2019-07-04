// compile-flags: --crate-type dylib --target thumbv7em-none-eabihf
// build-pass (FIXME(62277): could be check-pass?)
// error-pattern: dropping unsupported crate type `dylib` for target `thumbv7em-none-eabihf`

#![feature(no_core)]

#![no_std]
#![no_core]
