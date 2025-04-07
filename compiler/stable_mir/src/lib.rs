//! We've temporarily moved the `stable_mir` implementation to [`rustc_smir::stable_mir`],
//! during refactoring to break the circular dependency between `rustc_smir` and `stable_mir`,
//!
//! This is a transitional measure as described in [PR #139319](https://github.com/rust-lang/rust/pull/139319).
//! Once the refactoring is complete, the `stable_mir` implementation will be moved back here.

pub use rustc_smir::stable_mir::*;
