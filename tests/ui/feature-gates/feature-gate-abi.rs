// gate-test-intrinsics
//@ add-core-stubs
//@ compile-flags: --crate-type=rlib

#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

extern "rust-call" fn f4(_: ()) {} //~ ERROR extern "rust-call" ABI is experimental and subject to change

// Methods in trait definition
trait Tr {
    extern "rust-call" fn m4(_: ()); //~ ERROR extern "rust-call" ABI is experimental and subject to change

    extern "rust-call" fn dm4(_: ()) {} //~ ERROR extern "rust-call" ABI is experimental and subject to change
}

struct S;

// Methods in trait impl
impl Tr for S {
    extern "rust-call" fn m4(_: ()) {} //~ ERROR extern "rust-call" ABI is experimental and subject to change
}

// Methods in inherent impl
impl S {
    extern "rust-call" fn im4(_: ()) {} //~ ERROR extern "rust-call" ABI is experimental and subject to change
}

// Function pointer types
type A4 = extern "rust-call" fn(_: ()); //~ ERROR extern "rust-call" ABI is experimental and subject to change

// Foreign modules
extern "rust-call" {} //~ ERROR extern "rust-call" ABI is experimental and subject to change
