//! This crate contains [SyncTable] and [SyncPushVec] which offers lock-free reads and uses
//! quiescent state based reclamation for which an API is available in the [collect] module.

#![feature(
    alloc_layout_extra,
    allocator_api,
    core_intrinsics,
    dropck_eyepatch,
    extend_one,
    negative_impls,
    thread_local
)]
#![allow(clippy::len_without_is_empty, clippy::type_complexity)]

#[macro_use]
mod macros;

pub mod collect;
mod raw;
mod scopeguard;
mod util;

pub mod sync_push_vec;
pub mod sync_table;

pub use sync_push_vec::SyncPushVec;
pub use sync_table::SyncTable;
