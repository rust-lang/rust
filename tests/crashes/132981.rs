//@ known-bug: #132981
//@compile-flags: -Clink-dead-code=true --crate-type lib
//@ only-x86_64
//@ ignore-windows
// The set of targets this crashes on is really fiddly, because it is deep in our ABI logic. It
// crashes on x86_64-unknown-linux-gnu, and i686-pc-windows-msvc, but not on
// x86_64-pc-windows-msvc. If you are trying to fix this crash, don't pay too much attention to the
// directives.

#![feature(rust_cold_cc)]
pub extern "rust-cold" fn foo(_: [usize; 3]) {}
