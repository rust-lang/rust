//! Regression test for <https://github.com/rust-lang/rust/issues/141577>.
//!
//! The changes in <https://github.com/rust-lang/rust/pull/144298> exposed a
//! latent bug that would sometimes cause the compiler to emit a covfun record
//! for a function, but not emit a corresponding PGO symbol name entry, because
//! the function did not have any physical coverage counters. The `llvm-cov`
//! tool would then fail to resolve the covfun record's function name hash,
//! and exit with the cryptic error:
//!
//! ```text
//!    malformed instrumentation profile data: function name is empty
//! ```
//!
//! The bug was then triggered in the wild by the macro-expansion of
//! `#[derive(arbitrary::Arbitrary)]`.
//!
//! This test uses a minimized form of the `Arbitrary` derive macro that was
//! found to still trigger the original bug. The bug could also be triggered
//! by a bang proc-macro or an attribute proc-macro.

//@ edition: 2024
//@ revisions: attr bang derive
//@ proc-macro: try_in_macro_helper.rs

trait Arbitrary {
    fn try_size_hint() -> Option<usize>;
}

// Expand via an attribute proc-macro.
#[cfg_attr(attr, try_in_macro_helper::attr)]
const _: () = ();

// Expand via a regular bang-style proc-macro.
#[cfg(bang)]
try_in_macro_helper::bang!();

// Expand via a derive proc-macro.
#[cfg_attr(derive, derive(try_in_macro_helper::Arbitrary))]
enum MyEnum {}

fn main() {
    MyEnum::try_size_hint();
}
