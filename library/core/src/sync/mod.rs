//! Synchronization primitives

#![stable(feature = "rust1", since = "1.0.0")]
#![allow(safe_fn_direct_use_of_unsafe_op_on_args)]

pub mod atomic;
mod sync_view;
#[unstable(feature = "exclusive_wrapper", issue = "98407")]
pub use sync_view::SyncView;
