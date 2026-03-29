//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/154383>.
// Rustdoc used to ICE on the doc link attached to a `pub use` inside a
// private module.

mod inner {
    /// [std::vec::Vec]
    pub use std::vec::Vec as MyVec;
}
