use rustc_middle::mir::interpret::{InterpResult, Pointer};
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_target::abi::{Align, Size};

use super::util::ensure_monomorphic_enough;
use super::{InterpCx, Machine};

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    /// Creates a dynamic vtable for the given type and vtable origin. This is used only for
    /// objects.
    ///
    /// The `trait_ref` encodes the erased self type. Hence, if we are making an object `Foo<Trait>`
    /// from a value of type `Foo<T>`, then `trait_ref` would map `T: Trait`. `None` here means that
    /// this is an auto trait without any methods, so we only need the basic vtable (drop, size,
    /// align).
    pub fn get_vtable_ptr(
        &self,
        ty: Ty<'tcx>,
        poly_trait_ref: Option<ty::PolyExistentialTraitRef<'tcx>>,
    ) -> InterpResult<'tcx, Pointer<Option<M::Provenance>>> {
        trace!("get_vtable(trait_ref={:?})", poly_trait_ref);

        let (ty, poly_trait_ref) = self.tcx.erase_regions((ty, poly_trait_ref));

        // All vtables must be monomorphic, bail out otherwise.
        ensure_monomorphic_enough(*self.tcx, ty)?;
        ensure_monomorphic_enough(*self.tcx, poly_trait_ref)?;

        let vtable_symbolic_allocation = self.tcx.create_vtable_alloc(ty, poly_trait_ref);
        let vtable_ptr = self.global_base_pointer(Pointer::from(vtable_symbolic_allocation))?;
        Ok(vtable_ptr.into())
    }

    /// Returns a high-level representation of the entries of the given vtable.
    pub fn get_vtable_entries(
        &self,
        vtable: Pointer<Option<M::Provenance>>,
    ) -> InterpResult<'tcx, &'tcx [ty::VtblEntry<'tcx>]> {
        let (ty, poly_trait_ref) = self.get_ptr_vtable(vtable)?;
        Ok(if let Some(poly_trait_ref) = poly_trait_ref {
            let trait_ref = poly_trait_ref.with_self_ty(*self.tcx, ty);
            let trait_ref = self.tcx.erase_regions(trait_ref);
            self.tcx.vtable_entries(trait_ref)
        } else {
            TyCtxt::COMMON_VTABLE_ENTRIES
        })
    }

    pub fn get_vtable_size_and_align(
        &self,
        vtable: Pointer<Option<M::Provenance>>,
    ) -> InterpResult<'tcx, (Size, Align)> {
        let (ty, _trait_ref) = self.get_ptr_vtable(vtable)?;
        let layout = self.layout_of(ty)?;
        assert!(!layout.is_unsized(), "there are no vtables for unsized types");
        Ok((layout.size, layout.align.abi))
    }
}
