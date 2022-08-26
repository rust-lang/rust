use std::convert::TryFrom;
use std::fmt;

use crate::mir::interpret::{alloc_range, AllocId, Allocation, Pointer, Scalar};
use crate::ty::{self, Instance, PolyTraitRef, Ty, TyCtxt};
use rustc_ast::Mutability;

#[derive(Clone, Copy, PartialEq, HashStable)]
pub enum VtblEntry<'tcx> {
    /// destructor of this type (used in vtable header)
    MetadataDropInPlace,
    /// layout size of this type (used in vtable header)
    MetadataSize,
    /// layout align of this type (used in vtable header)
    MetadataAlign,
    /// non-dispatchable associated function that is excluded from trait object
    Vacant,
    /// dispatchable associated function
    Method(Instance<'tcx>),
    /// pointer to a separate supertrait vtable, can be used by trait upcasting coercion
    TraitVPtr(PolyTraitRef<'tcx>),
}

impl<'tcx> fmt::Debug for VtblEntry<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // We want to call `Display` on `Instance` and `PolyTraitRef`,
        // so we implement this manually.
        match self {
            VtblEntry::MetadataDropInPlace => write!(f, "MetadataDropInPlace"),
            VtblEntry::MetadataSize => write!(f, "MetadataSize"),
            VtblEntry::MetadataAlign => write!(f, "MetadataAlign"),
            VtblEntry::Vacant => write!(f, "Vacant"),
            VtblEntry::Method(instance) => write!(f, "Method({})", instance),
            VtblEntry::TraitVPtr(trait_ref) => write!(f, "TraitVPtr({})", trait_ref),
        }
    }
}

// Needs to be associated with the `'tcx` lifetime
impl<'tcx> TyCtxt<'tcx> {
    pub const COMMON_VTABLE_ENTRIES: &'tcx [VtblEntry<'tcx>] =
        &[VtblEntry::MetadataDropInPlace, VtblEntry::MetadataSize, VtblEntry::MetadataAlign];
}

pub const COMMON_VTABLE_ENTRIES_DROPINPLACE: usize = 0;
pub const COMMON_VTABLE_ENTRIES_SIZE: usize = 1;
pub const COMMON_VTABLE_ENTRIES_ALIGN: usize = 2;

/// Retrieves an allocation that represents the contents of a vtable.
/// Since this is a query, allocations are cached and not duplicated.
pub(super) fn vtable_allocation_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: (Ty<'tcx>, Option<ty::PolyExistentialTraitRef<'tcx>>),
) -> AllocId {
    let (ty, poly_trait_ref) = key;

    let vtable_entries = if let Some(poly_trait_ref) = poly_trait_ref {
        let trait_ref = poly_trait_ref.with_self_ty(tcx, ty);
        let trait_ref = tcx.erase_regions(trait_ref);

        tcx.vtable_entries(trait_ref)
    } else {
        TyCtxt::COMMON_VTABLE_ENTRIES
    };

    let layout = tcx
        .layout_of(ty::ParamEnv::reveal_all().and(ty))
        .expect("failed to build vtable representation");
    assert!(!layout.is_unsized(), "can't create a vtable for an unsized type");
    let size = layout.size.bytes();
    let align = layout.align.abi.bytes();

    let ptr_size = tcx.data_layout.pointer_size;
    let ptr_align = tcx.data_layout.pointer_align.abi;

    let vtable_size = ptr_size * u64::try_from(vtable_entries.len()).unwrap();
    let mut vtable = Allocation::uninit(vtable_size, ptr_align, /* panic_on_fail */ true).unwrap();

    // No need to do any alignment checks on the memory accesses below, because we know the
    // allocation is correctly aligned as we created it above. Also we're only offsetting by
    // multiples of `ptr_align`, which means that it will stay aligned to `ptr_align`.

    for (idx, entry) in vtable_entries.iter().enumerate() {
        let idx: u64 = u64::try_from(idx).unwrap();
        let scalar = match entry {
            VtblEntry::MetadataDropInPlace => {
                let instance = ty::Instance::resolve_drop_in_place(tcx, ty);
                let fn_alloc_id = tcx.create_fn_alloc(instance);
                let fn_ptr = Pointer::from(fn_alloc_id);
                Scalar::from_pointer(fn_ptr, &tcx)
            }
            VtblEntry::MetadataSize => Scalar::from_uint(size, ptr_size).into(),
            VtblEntry::MetadataAlign => Scalar::from_uint(align, ptr_size).into(),
            VtblEntry::Vacant => continue,
            VtblEntry::Method(instance) => {
                // Prepare the fn ptr we write into the vtable.
                let instance = instance.polymorphize(tcx);
                let fn_alloc_id = tcx.create_fn_alloc(instance);
                let fn_ptr = Pointer::from(fn_alloc_id);
                Scalar::from_pointer(fn_ptr, &tcx)
            }
            VtblEntry::TraitVPtr(trait_ref) => {
                let super_trait_ref = trait_ref
                    .map_bound(|trait_ref| ty::ExistentialTraitRef::erase_self_ty(tcx, trait_ref));
                let supertrait_alloc_id = tcx.vtable_allocation((ty, Some(super_trait_ref)));
                let vptr = Pointer::from(supertrait_alloc_id);
                Scalar::from_pointer(vptr, &tcx)
            }
        };
        vtable
            .write_scalar(&tcx, alloc_range(ptr_size * idx, ptr_size), scalar)
            .expect("failed to build vtable representation");
    }

    vtable.mutability = Mutability::Not;
    tcx.create_memory_alloc(tcx.intern_const_alloc(vtable))
}
