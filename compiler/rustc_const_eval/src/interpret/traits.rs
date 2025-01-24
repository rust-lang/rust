use rustc_abi::{Align, Size};
use rustc_middle::mir::interpret::{InterpResult, Pointer};
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, ExistentialPredicateStableCmpExt, Ty, TyCtxt, VtblEntry};
use tracing::trace;

use super::util::ensure_monomorphic_enough;
use super::{
    InterpCx, MPlaceTy, Machine, MemPlaceMeta, OffsetMode, Projectable, interp_ok, throw_ub,
};

impl<'tcx, M: Machine<'tcx>> InterpCx<'tcx, M> {
    /// Creates a dynamic vtable for the given type and vtable origin. This is used only for
    /// objects.
    ///
    /// The `dyn_ty` encodes the erased self type. Hence, if we are making an object
    /// `Foo<dyn Trait<Assoc = A> + Send>` from a value of type `Foo<T>`, then `dyn_ty`
    /// would be `Trait<Assoc = A> + Send`. If this list doesn't have a principal trait ref,
    /// we only need the basic vtable prefix (drop, size, align).
    pub fn get_vtable_ptr(
        &self,
        ty: Ty<'tcx>,
        dyn_ty: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    ) -> InterpResult<'tcx, Pointer<Option<M::Provenance>>> {
        trace!("get_vtable(ty={ty:?}, dyn_ty={dyn_ty:?})");

        let (ty, dyn_ty) = self.tcx.erase_regions((ty, dyn_ty));

        // All vtables must be monomorphic, bail out otherwise.
        ensure_monomorphic_enough(*self.tcx, ty)?;
        ensure_monomorphic_enough(*self.tcx, dyn_ty)?;

        let salt = M::get_global_alloc_salt(self, None);
        let vtable_symbolic_allocation = self.tcx.reserve_and_set_vtable_alloc(ty, dyn_ty, salt);
        let vtable_ptr = self.global_root_pointer(Pointer::from(vtable_symbolic_allocation))?;
        interp_ok(vtable_ptr.into())
    }

    pub fn get_vtable_size_and_align(
        &self,
        vtable: Pointer<Option<M::Provenance>>,
        expected_trait: Option<&'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>>,
    ) -> InterpResult<'tcx, (Size, Align)> {
        let ty = self.get_ptr_vtable_ty(vtable, expected_trait)?;
        let layout = self.layout_of(ty)?;
        assert!(layout.is_sized(), "there are no vtables for unsized types");
        interp_ok((layout.size, layout.align.abi))
    }

    pub(super) fn vtable_entries(
        &self,
        trait_: Option<ty::PolyExistentialTraitRef<'tcx>>,
        dyn_ty: Ty<'tcx>,
    ) -> &'tcx [VtblEntry<'tcx>] {
        if let Some(trait_) = trait_ {
            let trait_ref = trait_.with_self_ty(*self.tcx, dyn_ty);
            let trait_ref = self.tcx.erase_regions(trait_ref);
            self.tcx.vtable_entries(trait_ref)
        } else {
            TyCtxt::COMMON_VTABLE_ENTRIES
        }
    }

    /// Check that the given vtable trait is valid for a pointer/reference/place with the given
    /// expected trait type.
    pub(super) fn check_vtable_for_type(
        &self,
        vtable_dyn_type: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
        expected_dyn_type: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    ) -> InterpResult<'tcx> {
        // We check validity by comparing the lists of predicates for equality. We *could* instead
        // check that the dynamic type to which the vtable belongs satisfies all the expected
        // predicates, but that would likely be a lot slower and seems unnecessarily permissive.

        // FIXME: we are skipping auto traits for now, but might revisit this in the future.
        let mut sorted_vtable: Vec<_> = vtable_dyn_type.without_auto_traits().collect();
        let mut sorted_expected: Vec<_> = expected_dyn_type.without_auto_traits().collect();
        // `skip_binder` here is okay because `stable_cmp` doesn't look at binders
        sorted_vtable.sort_by(|a, b| a.skip_binder().stable_cmp(*self.tcx, &b.skip_binder()));
        sorted_vtable.dedup();
        sorted_expected.sort_by(|a, b| a.skip_binder().stable_cmp(*self.tcx, &b.skip_binder()));
        sorted_expected.dedup();

        if sorted_vtable.len() != sorted_expected.len() {
            throw_ub!(InvalidVTableTrait { vtable_dyn_type, expected_dyn_type });
        }

        for (a_pred, b_pred) in std::iter::zip(sorted_vtable, sorted_expected) {
            let is_eq = match (a_pred.skip_binder(), b_pred.skip_binder()) {
                (
                    ty::ExistentialPredicate::Trait(a_data),
                    ty::ExistentialPredicate::Trait(b_data),
                ) => self.eq_in_param_env(a_pred.rebind(a_data), b_pred.rebind(b_data)),

                (
                    ty::ExistentialPredicate::Projection(a_data),
                    ty::ExistentialPredicate::Projection(b_data),
                ) => self.eq_in_param_env(a_pred.rebind(a_data), b_pred.rebind(b_data)),

                _ => false,
            };
            if !is_eq {
                throw_ub!(InvalidVTableTrait { vtable_dyn_type, expected_dyn_type });
            }
        }

        interp_ok(())
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
        interp_ok(mplace)
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
        interp_ok(data)
    }
}
