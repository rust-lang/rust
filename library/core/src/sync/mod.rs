//! Synchronization primitives

#![stable(feature = "rust1", since = "1.0.0")]

pub mod atomic;
mod exclusive;
#[unstable(feature = "exclusive_wrapper", issue = "none")]
pub use exclusive::Exclusive;
