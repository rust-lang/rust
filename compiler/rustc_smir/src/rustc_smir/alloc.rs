use rustc_abi::{Align, Size};
use rustc_middle::mir::ConstValue;
use rustc_middle::mir::interpret::{AllocInit, AllocRange, Pointer, alloc_range};
use stable_mir::Error;
use stable_mir::mir::Mutability;
use stable_mir::ty::{Allocation, ProvenanceMap};

use crate::rustc_smir::{Stable, Tables};
use crate::stable_mir;

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
    const_value: ConstValue<'tcx>,
    tables: &mut Tables<'tcx>,
) -> Allocation {
    try_new_allocation(ty, const_value, tables)
        .unwrap_or_else(|_| panic!("Failed to convert: {const_value:?} to {ty:?}"))
}

#[allow(rustc::usage_of_qualified_ty)]
pub(crate) fn try_new_allocation<'tcx>(
    ty: rustc_middle::ty::Ty<'tcx>,
    const_value: ConstValue<'tcx>,
    tables: &mut Tables<'tcx>,
) -> Result<Allocation, Error> {
    let layout = tables
        .tcx
        .layout_of(rustc_middle::ty::TypingEnv::fully_monomorphized().as_query_input(ty))
        .map_err(|e| e.stable(tables))?;
    Ok(match const_value {
        ConstValue::Scalar(scalar) => {
            let size = scalar.size();
            let mut allocation = rustc_middle::mir::interpret::Allocation::new(
                size,
                layout.align.abi,
                AllocInit::Uninit,
                (),
            );
            allocation
                .write_scalar(&tables.tcx, alloc_range(Size::ZERO, size), scalar)
                .map_err(|e| e.stable(tables))?;
            allocation.stable(tables)
        }
        ConstValue::ZeroSized => new_empty_allocation(layout.align.abi),
        ConstValue::Slice { data, meta } => {
            let alloc_id = tables.tcx.reserve_and_set_memory_alloc(data);
            let ptr = Pointer::new(alloc_id.into(), Size::ZERO);
            let scalar_ptr = rustc_middle::mir::interpret::Scalar::from_pointer(ptr, &tables.tcx);
            let scalar_meta =
                rustc_middle::mir::interpret::Scalar::from_target_usize(meta, &tables.tcx);
            let mut allocation = rustc_middle::mir::interpret::Allocation::new(
                layout.size,
                layout.align.abi,
                AllocInit::Uninit,
                (),
            );
            allocation
                .write_scalar(
                    &tables.tcx,
                    alloc_range(Size::ZERO, tables.tcx.data_layout.pointer_size),
                    scalar_ptr,
                )
                .map_err(|e| e.stable(tables))?;
            allocation
                .write_scalar(
                    &tables.tcx,
                    alloc_range(tables.tcx.data_layout.pointer_size, scalar_meta.size()),
                    scalar_meta,
                )
                .map_err(|e| e.stable(tables))?;
            allocation.stable(tables)
        }
        ConstValue::Indirect { alloc_id, offset } => {
            let alloc = tables.tcx.global_alloc(alloc_id).unwrap_memory();
            allocation_filter(&alloc.0, alloc_range(offset, layout.size), tables)
        }
    })
}

/// Creates an `Allocation` only from information within the `AllocRange`.
pub(super) fn allocation_filter<'tcx>(
    alloc: &rustc_middle::mir::interpret::Allocation,
    alloc_range: AllocRange,
    tables: &mut Tables<'tcx>,
) -> Allocation {
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
        ptrs.push((
            offset.bytes_usize() - alloc_range.start.bytes_usize(),
            tables.prov(prov.alloc_id()),
        ));
    }
    Allocation {
        bytes,
        provenance: ProvenanceMap { ptrs },
        align: alloc.align.bytes(),
        mutability: alloc.mutability.stable(tables),
    }
}
