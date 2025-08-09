//! Memory allocation implementation for rustc_public.
//!
//! This module is responsible for constructing stable components.
//! All operations requiring rustc queries must be delegated
//! to `rustc_public_bridge::alloc` to maintain stability guarantees.

use rustc_abi::Align;
use rustc_middle::mir::ConstValue;
use rustc_middle::mir::interpret::AllocRange;
use rustc_public_bridge::bridge::Error as _;
use rustc_public_bridge::context::CompilerCtxt;
use rustc_public_bridge::{Tables, alloc};

use super::Error;
use super::compiler_interface::BridgeTys;
use super::mir::Mutability;
use super::ty::{Allocation, ProvenanceMap};
use super::unstable::Stable;

/// Creates new empty `Allocation` from given `Align`.
fn new_empty_allocation(align: Align) -> Allocation {
    Allocation {
        bytes: Vec::new(),
        provenance: ProvenanceMap { ptrs: Vec::new() },
        align: align.bytes(),
        mutability: Mutability::Not,
    }
}

// We need this method instead of a Stable implementation
// because we need to get `Ty` of the const we are trying to create, to do that
// we need to have access to `ConstantKind` but we can't access that inside Stable impl.
#[allow(rustc::usage_of_qualified_ty)]
pub(crate) fn new_allocation<'tcx>(
    ty: rustc_middle::ty::Ty<'tcx>,
    const_value: ConstValue,
    tables: &mut Tables<'tcx, BridgeTys>,
    cx: &CompilerCtxt<'tcx, BridgeTys>,
) -> Allocation {
    try_new_allocation(ty, const_value, tables, cx)
        .unwrap_or_else(|_| panic!("Failed to convert: {const_value:?} to {ty:?}"))
}

#[allow(rustc::usage_of_qualified_ty)]
pub(crate) fn try_new_allocation<'tcx>(
    ty: rustc_middle::ty::Ty<'tcx>,
    const_value: ConstValue,
    tables: &mut Tables<'tcx, BridgeTys>,
    cx: &CompilerCtxt<'tcx, BridgeTys>,
) -> Result<Allocation, Error> {
    let layout = alloc::create_ty_and_layout(cx, ty).map_err(|e| Error::from_internal(e))?;
    match const_value {
        ConstValue::Scalar(scalar) => {
            alloc::try_new_scalar(layout, scalar, cx).map(|alloc| alloc.stable(tables, cx))
        }
        ConstValue::ZeroSized => Ok(new_empty_allocation(layout.align.abi)),
        ConstValue::Slice { alloc_id, meta } => {
            alloc::try_new_slice(layout, alloc_id, meta, cx).map(|alloc| alloc.stable(tables, cx))
        }
        ConstValue::Indirect { alloc_id, offset } => {
            let alloc = alloc::try_new_indirect(alloc_id, cx);
            use rustc_public_bridge::context::AllocRangeHelpers;
            Ok(allocation_filter(&alloc.0, cx.alloc_range(offset, layout.size), tables, cx))
        }
    }
}

/// Creates an `Allocation` only from information within the `AllocRange`.
pub(super) fn allocation_filter<'tcx>(
    alloc: &rustc_middle::mir::interpret::Allocation,
    alloc_range: AllocRange,
    tables: &mut Tables<'tcx, BridgeTys>,
    cx: &CompilerCtxt<'tcx, BridgeTys>,
) -> Allocation {
    alloc::allocation_filter(alloc, alloc_range, tables, cx)
}
