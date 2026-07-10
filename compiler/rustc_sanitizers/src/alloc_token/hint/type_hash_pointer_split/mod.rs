//! Allocation token hints (using the pointer-split and type-hash-pointer-split heap partitioning
//! schemes) for LLVM AllocToken and heap partitioning support.
//!
//! For more information about LLVM AllocToken and heap partitioning support for the Rust compiler,
//! see design document in the tracking issue #159111.

use rustc_middle::ty::{Ty, TyCtxt};
use tracing::instrument;

use crate::alloc_token::hint::{AllocTokenHint, AllocTokenHintOptions};

mod classify;
mod encode;

/// Returns an allocation token hint for the given type using the pointer-split or
/// type-hash-pointer-split heap partitioning scheme.
#[instrument(level = "trace", skip(tcx))]
pub(super) fn hint_for_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    _options: AllocTokenHintOptions,
) -> AllocTokenHint {
    AllocTokenHint {
        type_name: encode::encode_ty(tcx, ty),
        contains_pointer: classify::contains_pointer(tcx, ty),
    }
}
