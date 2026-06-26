//@ check-pass
//@ compile-flags:--test --test-args=--test-threads=1
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"

// This test ensures that crate imports are placed outside of the `main` function
// so they work all the time (even in 2015 edition).

/// ```rust
/// #![feature(test)]
///
/// extern crate test;
/// use test::Bencher;
///
/// #[bench]
/// fn bench_xor_1000_ints(b: &mut Bencher) {
///     b.iter(|| {
///         (0..1000).fold(0, |old, new| old ^ new);
///     });
/// }
/// ```
///
pub fn foo() {}
