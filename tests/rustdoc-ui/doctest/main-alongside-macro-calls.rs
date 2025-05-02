// This test ensures that if there is are any macro calls alongside a `main` function,
// it will indeed consider the `main` function as the program entry point and *won't*
// generate its own `main` function to wrap everything even though macro calls are
// valid in statement contexts, too, and could just as well expand to statements or
// expressions (we don't perform any macro expansion to find `main`, see also
// <https://github.com/rust-lang/rust/issues/57415>).
//
// See <./main-alongside-stmts.rs> for comparison.
//
//@ compile-flags:--test --test-args --test-threads=1
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ revisions: pass fail
//@[pass] check-pass
//@[fail] failure-status: 101

// Regression test for <https://github.com/rust-lang/rust/pull/140220#issuecomment-2831872920>:

//! ```
//! fn main() {}
//! include!("./auxiliary/items.rs");
//! ```
//!
//! ```
//! include!("./auxiliary/items.rs");
//! fn main() {}
//! ```

// Regression test for <https://github.com/rust-lang/rust/issues/140412>:
// We test the "same" thing twice: Once via `compile_fail` to more closely mirror the reported
// regression and once without it to make sure that it leads to the expected rustc errors,
// namely `println!(…)` not being valid in item contexts.

#![cfg_attr(pass, doc = " ```compile_fail")]
#![cfg_attr(fail, doc = " ```")]
//! fn main() {}
//! println!();
//! ```
//!
#![cfg_attr(pass, doc = " ```compile_fail")]
#![cfg_attr(fail, doc = " ```")]
//! println!();
//! fn main() {}
//! ```
