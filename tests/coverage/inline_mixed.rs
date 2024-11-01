//@ edition: 2021
//@ compile-flags: -Cinstrument-coverage=off
//@ aux-crate: inline_mixed_helper=inline_mixed_helper.rs

// Regression test for <https://github.com/rust-lang/rust/pull/132395>.
// Various forms of cross-crate inlining can cause coverage statements to be
// inlined into crates that are being built without coverage instrumentation.
// At the very least, we need to not ICE when that happens.

fn main() {
    inline_mixed_helper::inline_me();
    inline_mixed_helper::no_inlining_please();
    inline_mixed_helper::generic::<u32>();
}
