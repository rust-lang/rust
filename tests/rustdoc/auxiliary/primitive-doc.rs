// compile-flags: --crate-type lib --edition 2018

#![feature(no_core)]
#![no_core]

#[doc(primitive = "usize")]
/// This is the built-in type `usize`.
mod usize {
}
