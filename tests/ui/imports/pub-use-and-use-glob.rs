//! Regression test for <https://github.com/rust-lang/rust/issues/151124>
//@ check-pass
mod bar {
    pub struct Symbol0;
}
use bar::*;
pub use bar::*;
fn main() {}
