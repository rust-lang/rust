//! Internal memory allocator implementation for StableMIR.
//!
//! This module handles all direct interactions with rustc queries and performs
//! the actual memory allocations. The stable interface in `stable_mir::alloc`
//! delegates all query-related operations to this implementation.

use rustc_abi::{Size, TyAndLayout};
use rustc_middle::mir::interpret::{
    AllocId, AllocInit, AllocRange, Allocation, ConstAllocation, Pointer, Scalar, alloc_range,
};
use rustc_middle::ty::Ty;

use crate::rustc_smir::{Bridge, SmirError, Tables};

pub fn try_new_scalar<'tcx, B: Bridge>(
    layout: TyAndLayout<'tcx, Ty<'tcx>>,
    scalar: Scalar,
    tables: &mut Tables<'tcx, B>,
) -> Result<Allocation, B::Error> {
    let size = scalar.size();
    let mut allocation = Allocation::new(size, layout.align.abi, AllocInit::Uninit, ());
    allocation
        .write_scalar(&tables.tcx, alloc_range(Size::ZERO, size), scalar)
        .map_err(|e| B::Error::from_internal(e))?;

    Ok(allocation)
}

pub fn try_new_slice<'tcx, B: Bridge>(
    layout: TyAndLayout<'tcx, Ty<'tcx>>,
    data: ConstAllocation<'tcx>,
    meta: u64,
    tables: &mut Tables<'tcx, B>,
) -> Result<Allocation, B::Error> {
    let alloc_id = tables.tcx.reserve_and_set_memory_alloc(data);
    let ptr = Pointer::new(alloc_id.into(), Size::ZERO);
    let scalar_ptr = Scalar::from_pointer(ptr, &tables.tcx);
    let scalar_meta: Scalar = Scalar::from_target_usize(meta, &tables.tcx);
    let mut allocation = Allocation::new(layout.size, layout.align.abi, AllocInit::Uninit, ());
    allocation
        .write_scalar(
            &tables.tcx,
            alloc_range(Size::ZERO, tables.tcx.data_layout.pointer_size),
            scalar_ptr,
        )
        .map_err(|e| B::Error::from_internal(e))?;
    allocation
        .write_scalar(
            &tables.tcx,
            alloc_range(tables.tcx.data_layout.pointer_size, scalar_meta.size()),
            scalar_meta,
        )
        .map_err(|e| B::Error::from_internal(e))?;

    Ok(allocation)
}

pub fn try_new_indirect<'tcx, B: Bridge>(
    alloc_id: AllocId,
    tables: &mut Tables<'tcx, B>,
) -> ConstAllocation<'tcx> {
    let alloc = tables.tcx.global_alloc(alloc_id).unwrap_memory();

    alloc
}

/// Creates an `Allocation` only from information within the `AllocRange`.
pub fn allocation_filter(
    alloc: &rustc_middle::mir::interpret::Allocation,
    alloc_range: AllocRange,
) -> (Vec<Option<u8>>, Vec<(usize, AllocId)>) {
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

    (bytes, ptrs)
}
