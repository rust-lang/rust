// Regression test for <https://github.com/rust-lang/rust/issues/109282>.
// Import for `ValueEnum` is inlined and doc comments on the import and `ValueEnum` itself are
// merged. After the merge they still have correct parent scopes to resolve both `[ValueEnum]`.

//@ check-pass

mod m {
    pub enum ValueEnum {}
}
mod m2 {
    /// [`ValueEnum`]
    pub use crate::m::ValueEnum;
}
pub use m2::ValueEnum;
