//! Synchronization primitives

#![stable(feature = "rust1", since = "1.0.0")]

pub mod atomic;
mod sync_view;
#[stable(feature = "exclusive_wrapper", since = "1.101.0")]
pub use sync_view::SyncView;
