//! Synchronization primitives

#![stable(feature = "rust1", since = "1.0.0")]

pub mod atomic;
mod sync_view;
#[unstable(feature = "exclusive_wrapper", issue = "98407")]
pub use sync_view::SyncView;
