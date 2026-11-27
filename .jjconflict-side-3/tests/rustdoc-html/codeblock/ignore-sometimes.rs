#![crate_name = "foo"]

// test for https://github.com/rust-lang/rust/issues/141092

//@ has 'foo/fn.f.html' '//a[@title="This example is not tested on wasm"]' 'ⓘ'
/// Example
///
/// ```ignore-wasm
/// let x = 1;
/// ```
pub fn f() {}

//@ has 'foo/fn.g.html' '//a[@title="This example is not tested on wasm or windows"]' 'ⓘ'
/// ```ignore-wasm,ignore-windows
/// let x = 1;
/// ```
pub fn g() {}

//@ has 'foo/fn.h.html' '//a[@title="This example is not tested on wasm, windows, or unix"]' 'ⓘ'
/// ```ignore-wasm,ignore-windows,ignore-unix
/// let x = 1;
/// ```
pub fn h() {}
