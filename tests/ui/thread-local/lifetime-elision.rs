// Test that  `thread_local!` defaults elided lifetimes in the type to `'static`.
// Regression test for 159358.
// Ideally this should be tested at least one target using each `thread_local!`
// implementation (no-threads, native, os), but currently there are no Tier 1 `no-threads`
// targets. `x86_64-unknown-linux-gnu` is a `native` target and `x86_64-pc-windows-gnu`
// is a `os` target, so those are covered (as of 2026-07-21).
//@ check-pass

// Const initializer
std::thread_local!(static A: &str = const { "" });
// Non-const initializer
std::thread_local!(static B: &str = "");

fn main() {}
