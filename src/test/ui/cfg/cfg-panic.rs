// build-pass
// compile-flags: -C panic=unwind
// needs-unwind
// ignore-emscripten no panic_unwind implementation
// ignore-wasm32     no panic_unwind implementation
// ignore-wasm64     no panic_unwind implementation


#[cfg(panic = "abort")]
pub fn bad() -> i32 { }

#[cfg(not(panic = "unwind"))]
pub fn bad() -> i32 { }

#[cfg(panic = "some_imaginary_future_panic_handler")]
pub fn bad() -> i32 { }

#[cfg(panic = "unwind")]
pub fn main() { }
