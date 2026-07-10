//! Allocation token hints for LLVM AllocToken and heap partitioning support for the Rust compiler.
//!
//! For more information about LLVM AllocToken and heap partitioning support for the Rust compiler,
//! see design document in the tracking issue #159111.

use bitflags::bitflags;
use rustc_middle::ty::{Ty, TyCtxt};
use tracing::instrument;

bitflags! {
    /// Options for hint_for_ty.
    #[derive(Clone, Copy, Debug)]
    pub struct AllocTokenHintOptions: u32 {
    }
}

/// An allocation token hint (i.e., the contents of the `!alloc_token` metadata) for a given type.
#[derive(Debug)]
pub struct AllocTokenHint {
    /// A type name uniquely identifying the allocated type, used by the TypeHash and
    /// TypeHashPointerSplit modes to derive stable token identifiers.
    pub type_name: String,
    /// Whether the allocated type contains pointers, used by the TypeHashPointerSplit mode to
    /// partition the token identifier space.
    pub contains_pointer: bool,
}

pub mod type_hash_pointer_split;

/// Returns an allocation token hint for the given type, computed by the selected heap
/// partitioning scheme.
#[instrument(level = "trace", skip(tcx))]
pub fn hint_for_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    options: AllocTokenHintOptions,
) -> AllocTokenHint {
    type_hash_pointer_split::hint_for_ty(tcx, ty, options)
}

/// Returns a conservative allocation token hint for an allocation whose type is unknown (e.g.,
/// for calls to allocation functions reached through type-erased paths), so that default or
/// unknown is the partition containing pointers.
pub fn hint_for_unknown_ty() -> AllocTokenHint {
    AllocTokenHint { type_name: String::new(), contains_pointer: true }
}
