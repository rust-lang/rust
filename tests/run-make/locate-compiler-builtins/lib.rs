#![no_std]
#![no_implicit_prelude]
#![crate_type = "dylib"]
// Hack: `compiler_builtins` is not injected in crates with `#![no_core]`.
// However, we still resolve to the local core since sysroot is
// disabled in this test.
