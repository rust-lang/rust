// compile-flags:-Zprint-mono-items=lazy

// rust-lang/rust#90405
// Ensure implicit panic calls are collected

#![feature(lang_items)]
#![feature(no_core)]
#![crate_type = "lib"]
#![no_core]
#![no_std]

#[lang = "panic_location"]
struct Location<'a> {
    _file: &'a str,
    _line: u32,
    _col: u32,
}

#[lang = "panic"]
#[inline]
#[track_caller]
fn panic(_: &'static str) -> ! {
    loop {}
}

#[lang = "sized"]
trait Sized {}

#[lang = "copy"]
trait Copy {}

#[lang = "freeze"]
trait Freeze {}

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
//~ MONO_ITEM fn panic
