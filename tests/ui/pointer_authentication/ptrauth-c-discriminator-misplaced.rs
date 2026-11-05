//@ add-minicore
//@ build-fail
//@ compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination
//@ needs-llvm-components: aarch64
//@ ignore-backends: gcc
// ignore-tidy-linelength

// `#[ptrauth_c_discriminator(N)]` marks a slot: a pointer, a pointer-sized integer, or an array of
// them.
// Error out if it is used on:
// * a static of any other type,
// * on a field of any other type,
// * on overlapping union fields with different discriminators.

#![feature(no_core, ptrauth_c_discriminator)]
#![no_std]
#![no_core]
#![allow(dead_code)]

extern crate minicore;
use minicore::*;

extern "C" fn f() {}

#[repr(C)]
pub struct Pair {
    a: *const u8,
    b: *const u8,
}

#[used]
#[ptrauth_c_discriminator(0)]
static mut PAIR: Pair = Pair { a: f as *const u8, b: 0 as *const u8 };
//~^ ERROR `#[ptrauth_c_discriminator]` on a static needs a pointer or a pointer-sized integer, or an array of them, not `Pair`

pub struct Small {
    #[ptrauth_c_discriminator(0)]
    pub tag: u8,
    //~^ ERROR `#[ptrauth_c_discriminator]` needs a field that is a pointer or a pointer-sized integer, or an array of them, not `u8`
}

#[used]
static mut SMALL: Small = Small { tag: 1 };

#[repr(C)]
pub union Overlap {
    #[ptrauth_c_discriminator(0)]
    pub a: *const u8,
    #[ptrauth_c_discriminator(1)]
    pub b: *const u8,
}

#[used]
static mut OVERLAP: Overlap = Overlap { a: f as *const u8 };
//~^ ERROR fields marked `#[ptrauth_c_discriminator]` with different discriminators overlap at offset 0
