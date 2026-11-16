//@ check-pass
// NOTE: This test requires a feature gated const fn and will stop working in the future.
// NOTE: See https://github.com/rust-lang/rust/issues/164718 for details on a permanent fix.

#![feature(const_unsigned_bigint_helpers)]

fn test() -> (u32, bool) {
    u32::carrying_add(10, 1, true)
}

fn main() {}
