use rustc_middle::mir::interpret::{InterpResult, Pointer};
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, Ty};
use rustc_target::abi::{Align, Size};
use tracing::trace;

use super::util::ensure_monomorphic_enough;
use super::{InterpCx, MPlaceTy, Machine, MemPlaceMeta, OffsetMode, Projectable};

impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
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

        let vtable_symbolic_allocation = self.tcx.reserve_and_set_vtable_alloc(ty, poly_trait_ref);
        let vtable_ptr = self.global_root_pointer(Pointer::from(vtable_symbolic_allocation))?;
        Ok(vtable_ptr.into())
    }

    pub fn get_vtable_size_and_align(
        &self,
        vtable: Pointer<Option<M::Provenance>>,
    ) -> InterpResult<'tcx, (Size, Align)> {
        let ty = self.get_ptr_vtable_ty(vtable, None)?;
        let layout = self.layout_of(ty)?;
        assert!(layout.is_sized(), "there are no vtables for unsized types");
        Ok((layout.size, layout.align.abi))
    }

    /// Turn a place with a `dyn Trait` type into a place with the actual dynamic type.
    pub(super) fn unpack_dyn_trait(
        &self,
        mplace: &MPlaceTy<'tcx, M::Provenance>,
        expected_trait: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    ) -> InterpResult<'tcx, MPlaceTy<'tcx, M::Provenance>> {
        assert!(
            matches!(mplace.layout.ty.kind(), ty::Dynamic(_, _, ty::Dyn)),
            "`unpack_dyn_trait` only makes sense on `dyn*` types"
        );
        let vtable = mplace.meta().unwrap_meta().to_pointer(self)?;
        let ty = self.get_ptr_vtable_ty(vtable, Some(expected_trait))?;
        // This is a kind of transmute, from a place with unsized type and metadata to
        // a place with sized type and no metadata.
        let layout = self.layout_of(ty)?;
        let mplace = mplace.offset_with_meta(
            Size::ZERO,
            OffsetMode::Wrapping,
            MemPlaceMeta::None,
            layout,
            self,
        )?;
        Ok(mplace)
    }

    /// Turn a `dyn* Trait` type into an value with the actual dynamic type.
    pub(super) fn unpack_dyn_star<P: Projectable<'tcx, M::Provenance>>(
        &self,
        val: &P,
        expected_trait: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    ) -> InterpResult<'tcx, P> {
        assert!(
            matches!(val.layout().ty.kind(), ty::Dynamic(_, _, ty::DynStar)),
            "`unpack_dyn_star` only makes sense on `dyn*` types"
        );
        let data = self.project_field(val, 0)?;
        let vtable = self.project_field(val, 1)?;
        let vtable = self.read_pointer(&vtable.to_op(self)?)?;
        let ty = self.get_ptr_vtable_ty(vtable, Some(expected_trait))?;
        // `data` is already the right thing but has the wrong type. So we transmute it.
        let layout = self.layout_of(ty)?;
        let data = data.transmute(layout, self)?;
        Ok(data)
    }
}
