// compile-flags: --crate-type lib --edition 2018
#![no_core]
#![feature(no_core)]
#![feature(lang_items)]

#[cfg(windows)]
#[no_mangle]
extern "system" fn _DllMainCRTStartup(_: *const u8, _: u32, _: *const u8) -> u32 { 1 }

#[cfg(windows)]
#[lang = "sized"]
trait Sized {}

#[doc(primitive = "usize")]
/// This is the built-in type `usize`.
mod usize {
}
