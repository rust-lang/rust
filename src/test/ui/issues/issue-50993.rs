// compile-flags: --crate-type dylib --target thumbv7em-none-eabihf
// build-pass
// error-pattern: dropping unsupported crate type `dylib` for target `thumbv7em-none-eabihf`

// ignore-emscripten FIXME: debugging only, do not land

#![feature(no_core)]

#![no_std]
#![no_core]
