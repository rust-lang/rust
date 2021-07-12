use std::convert::TryFrom;

use crate::mir::interpret::{alloc_range, AllocId, Allocation, Pointer, Scalar, ScalarMaybeUninit};
use crate::ty::fold::TypeFoldable;
use crate::ty::{self, DefId, SubstsRef, Ty, TyCtxt};
use rustc_ast::Mutability;

#[derive(Clone, Copy, Debug, PartialEq, HashStable)]
pub enum VtblEntry<'tcx> {
    MetadataDropInPlace,
    MetadataSize,
    MetadataAlign,
    Vacant,
    Method(DefId, SubstsRef<'tcx>),
}

pub const COMMON_VTABLE_ENTRIES: &[VtblEntry<'_>] =
    &[VtblEntry::MetadataDropInPlace, VtblEntry::MetadataSize, VtblEntry::MetadataAlign];

pub const COMMON_VTABLE_ENTRIES_DROPINPLACE: usize = 0;
pub const COMMON_VTABLE_ENTRIES_SIZE: usize = 1;
pub const COMMON_VTABLE_ENTRIES_ALIGN: usize = 2;

impl<'tcx> TyCtxt<'tcx> {
    /// Retrieves an allocation that represents the contents of a vtable.
    /// There's a cache within `TyCtxt` so it will be deduplicated.
    pub fn vtable_allocation(
        self,
        ty: Ty<'tcx>,
        poly_trait_ref: Option<ty::PolyExistentialTraitRef<'tcx>>,
    ) -> AllocId {
        let tcx = self;
        let vtables_cache = tcx.vtables_cache.lock();
        if let Some(alloc_id) = vtables_cache.get(&(ty, poly_trait_ref)).cloned() {
            return alloc_id;
        }
        drop(vtables_cache);

        // See https://github.com/rust-lang/rust/pull/86475#discussion_r655162674
        assert!(
            !ty.needs_subst() && !poly_trait_ref.map_or(false, |trait_ref| trait_ref.needs_subst())
        );
        let param_env = ty::ParamEnv::reveal_all();
        let vtable_entries = if let Some(poly_trait_ref) = poly_trait_ref {
            let trait_ref = poly_trait_ref.with_self_ty(tcx, ty);
            let trait_ref = tcx.erase_regions(trait_ref);

            tcx.vtable_entries(trait_ref)
        } else {
            COMMON_VTABLE_ENTRIES
        };

        let layout =
            tcx.layout_of(param_env.and(ty)).expect("failed to build vtable representation");
        assert!(!layout.is_unsized(), "can't create a vtable for an unsized type");
        let size = layout.size.bytes();
        let align = layout.align.abi.bytes();

        let ptr_size = tcx.data_layout.pointer_size;
        let ptr_align = tcx.data_layout.pointer_align.abi;

        let vtable_size = ptr_size * u64::try_from(vtable_entries.len()).unwrap();
        let mut vtable =
            Allocation::uninit(vtable_size, ptr_align, /* panic_on_fail */ true).unwrap();

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
                    ScalarMaybeUninit::from_pointer(fn_ptr, &tcx)
                }
                VtblEntry::MetadataSize => Scalar::from_uint(size, ptr_size).into(),
                VtblEntry::MetadataAlign => Scalar::from_uint(align, ptr_size).into(),
                VtblEntry::Vacant => continue,
                VtblEntry::Method(def_id, substs) => {
                    // See https://github.com/rust-lang/rust/pull/86475#discussion_r655162674
                    assert!(!substs.needs_subst());

                    // Prepare the fn ptr we write into the vtable.
                    let instance =
                        ty::Instance::resolve_for_vtable(tcx, param_env, *def_id, substs)
                            .expect("resolution failed during building vtable representation")
                            .polymorphize(tcx);
                    let fn_alloc_id = tcx.create_fn_alloc(instance);
                    let fn_ptr = Pointer::from(fn_alloc_id);
                    ScalarMaybeUninit::from_pointer(fn_ptr, &tcx)
                }
            };
            vtable
                .write_scalar(&tcx, alloc_range(ptr_size * idx, ptr_size), scalar)
                .expect("failed to build vtable representation");
        }

        vtable.mutability = Mutability::Not;
        let alloc_id = tcx.create_memory_alloc(tcx.intern_const_alloc(vtable));
        let mut vtables_cache = self.vtables_cache.lock();
        vtables_cache.insert((ty, poly_trait_ref), alloc_id);
        alloc_id
    }
}
