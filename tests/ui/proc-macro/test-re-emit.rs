//@ check-pass
//@ proc-macro: test-re-emit.rs
//@ compile-flags: --test
// Test that we can pass a test through a proc macro that removes the span of the item
// Regression test for https://github.com/rust-lang/rust/issues/161917

#[test]
#[test_re_emit::remove_span]
fn meow1() {}

#[test_re_emit::remove_span]
#[test]
fn meow2() {}

fn main() {}
