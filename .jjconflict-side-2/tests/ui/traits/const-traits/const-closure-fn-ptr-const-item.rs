//@ run-pass

// Regression test for https://github.com/rust-lang/rust/issues/155803

#![feature(const_closures, const_trait_impl)]

const F: fn() -> i32 = const || 42;

fn main() {
    assert_eq!(F(), 42);
}
