// compile-flags: --test --crate-name=empty_fn --extern=empty_fn --test-args=--test-threads=1
// aux-build:empty-fn.rs
// check-pass
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"
// normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"
// edition:2021

/// <https://github.com/rust-lang/rust/issues/91134>
///
/// ```
/// extern crate empty_fn;
/// empty_fn::empty();
/// ```
pub struct Something;
