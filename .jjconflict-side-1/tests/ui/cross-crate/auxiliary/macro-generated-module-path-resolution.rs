//! Auxiliary crate testing this issue https://github.com/rust-lang/rust/issues/38190
#[macro_export]
macro_rules! m { ([$i:item]) => {} }
