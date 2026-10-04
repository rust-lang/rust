// `-z text` caused the linker to error if there were any non-position-independent
// code (PIC) sections. This test checks that this no longer happens.
// See https://github.com/rust-lang/rust/pull/39803

//@ ignore-windows
//@ ignore-macos
//@ ignore-cross-compile
//@ ignore-aix

//@ compile-flags: -Clink-args=-Wl,-z,text
//@ run-pass

fn main() {}
