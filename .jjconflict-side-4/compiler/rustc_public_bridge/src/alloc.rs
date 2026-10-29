//! Internal memory allocator implementation for rustc_public.
//!
//! This module handles all direct interactions with rustc queries and performs
//! the actual memory allocations. The stable interface in `rustc_public::alloc`
//! delegates all query-related operations to this implementation.

use rustc_abi::{Size, TyAndLayout};
use rustc_middle::mir::interpret::{
    AllocId, AllocInit, AllocRange, Allocation, ConstAllocation, Pointer, Scalar, alloc_range,
};
use rustc_middle::ty::{Ty, layout};

use super::{CompilerCtxt, Tables};
use crate::bridge::Allocation as _;
use crate::{Bridge, Error};

pub fn create_ty_and_layout<'tcx, B: Bridge>(
    cx: &CompilerCtxt<'tcx, B>,
    ty: Ty<'tcx>,
) -> Result<TyAndLayout<'tcx, Ty<'tcx>>, &'tcx layout::LayoutError<'tcx>> {
    use crate::context::TypingEnvHelpers;
    cx.tcx.layout_of(cx.fully_monomorphized().as_query_input(ty))
}

pub fn try_new_scalar<'tcx, B: Bridge>(
    layout: TyAndLayout<'tcx, Ty<'tcx>>,
    scalar: Scalar,
    cx: &CompilerCtxt<'tcx, B>,
) -> Result<Allocation, B::Error> {
    let size = scalar.size();
    let mut allocation = Allocation::new(size, layout.align.abi, AllocInit::Uninit, ());
    allocation
        .write_scalar(&cx.tcx, alloc_range(Size::ZERO, size), scalar)
        .map_err(|e| B::Error::from_internal(e))?;

    Ok(allocation)
}

pub fn try_new_slice<'tcx, B: Bridge>(
    layout: TyAndLayout<'tcx, Ty<'tcx>>,
    alloc_id: AllocId,
    meta: u64,
    cx: &CompilerCtxt<'tcx, B>,
) -> Result<Allocation, B::Error> {
    let ptr = Pointer::new(alloc_id.into(), Size::ZERO);
    let scalar_ptr = Scalar::from_pointer(ptr, &cx.tcx);
    let scalar_meta: Scalar = Scalar::from_target_usize(meta, &cx.tcx);
    let mut allocation = Allocation::new(layout.size, layout.align.abi, AllocInit::Uninit, ());
    let ptr_size = cx.tcx.data_layout.pointer_size();
    allocation
        .write_scalar(&cx.tcx, alloc_range(Size::ZERO, ptr_size), scalar_ptr)
        .map_err(|e| B::Error::from_internal(e))?;
    allocation
        .write_scalar(&cx.tcx, alloc_range(ptr_size, scalar_meta.size()), scalar_meta)
        .map_err(|e| B::Error::from_internal(e))?;

    Ok(allocation)
}

pub fn try_new_indirect<'tcx, B: Bridge>(
    alloc_id: AllocId,
    cx: &CompilerCtxt<'tcx, B>,
) -> ConstAllocation<'tcx> {
    let alloc = cx.tcx.global_alloc(alloc_id).unwrap_memory();

    alloc
}

/// Creates an `Allocation` only from information within the `AllocRange`.
pub fn allocation_filter<'tcx, B: Bridge>(
    alloc: &rustc_middle::mir::interpret::Allocation,
    alloc_range: AllocRange,
    tables: &mut Tables<'tcx, B>,
    cx: &CompilerCtxt<'tcx, B>,
) -> B::Allocation {
    let mut bytes: Vec<Option<u8>> = alloc
        .inspect_with_uninit_and_ptr_outside_interpreter(
            alloc_range.start.bytes_usize()..alloc_range.end().bytes_usize(),
        )
        .iter()
        .copied()
        .map(Some)
        .collect();
    for (i, b) in bytes.iter_mut().enumerate() {
        if !alloc.init_mask().get(Size::from_bytes(i + alloc_range.start.bytes_usize())) {
            *b = None;
        }
    }
    let mut ptrs = Vec::new();
    for (offset, prov) in alloc
        .provenance()
        .ptrs()
        .iter()
        .filter(|a| a.0 >= alloc_range.start && a.0 <= alloc_range.end())
    {
        ptrs.push((offset.bytes_usize() - alloc_range.start.bytes_usize(), prov.alloc_id()));
    }

    B::Allocation::new(bytes, ptrs, alloc.align.bytes(), alloc.mutability, tables, cx)
}
