// Regression test for <https://github.com/rust-lang/rust/pull/113167>

//@ check-pass
#![deny(rustdoc::redundant_explicit_links)]

mod m {
    pub enum ValueEnum {}
}
mod m2 {
    /// [`ValueEnum`]
    pub use crate::m::ValueEnum;
}
pub use m2::ValueEnum;
