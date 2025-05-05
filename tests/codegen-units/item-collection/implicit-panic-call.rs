//@ add-core-stubs
// @ compile-flags:-Zprint-mono-items=lazy

// rust-lang/rust#90405
// Ensure implicit panic calls are collected

#![feature(lang_items)]
#![feature(no_core)]
#![crate_type = "lib"]
#![no_core]
#![no_std]

extern crate minicore;
use minicore::*;

impl Copy for i32 {}

#[lang = "div"]
trait Div<Rhs = Self> {
    type Output;
    fn div(self, rhs: Rhs) -> Self::Output;
}

impl Div for i32 {
    type Output = i32;
    fn div(self, rhs: i32) -> i32 {
        self / rhs
    }
}

#[allow(unconditional_panic)]
pub fn foo() {
    // This implicitly generates a panic call.
    let _ = 1 / 0;
}

//~ MONO_ITEM fn foo
//~ MONO_ITEM fn <i32 as Div>::div
//~ MONO_ITEM fn panic_div_zero
//~ MONO_ITEM fn panic_div_overflow
