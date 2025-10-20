//! Definition of `InferCtxtLike` from the librarified type layer.

use rustc_type_ir::{
    ConstVid, FloatVarValue, FloatVid, GenericArgKind, InferConst, InferTy, IntVarValue, IntVid,
    RegionVid, TyVid, TypeFoldable, TypingMode, UniverseIndex,
    inherent::{Const as _, IntoKind, Ty as _},
    relate::combine::PredicateEmittingRelation,
};

use crate::next_solver::{
    Binder, Const, ConstKind, DbInterner, ErrorGuaranteed, GenericArgs, OpaqueTypeKey, Region,
    SolverDefId, Span, Ty, TyKind,
    infer::opaque_types::{OpaqueHiddenType, table::OpaqueTypeStorageEntries},
};

use super::{BoundRegionConversionTime, InferCtxt, relate::RelateResult};

impl<'db> rustc_type_ir::InferCtxtLike for InferCtxt<'db> {
    type Interner = DbInterner<'db>;

    fn cx(&self) -> DbInterner<'db> {
        self.interner
    }

    fn next_trait_solver(&self) -> bool {
        true
    }

    fn typing_mode(&self) -> TypingMode<DbInterner<'db>> {
        self.typing_mode()
    }

    fn universe(&self) -> UniverseIndex {
        self.universe()
    }

    fn create_next_universe(&self) -> UniverseIndex {
        self.create_next_universe()
    }

    fn universe_of_ty(&self, vid: TyVid) -> Option<UniverseIndex> {
        self.probe_ty_var(vid).err()
    }

    fn universe_of_lt(&self, lt: RegionVid) -> Option<UniverseIndex> {
        self.inner.borrow_mut().unwrap_region_constraints().probe_value(lt).err()
    }

    fn universe_of_ct(&self, ct: ConstVid) -> Option<UniverseIndex> {
        self.probe_const_var(ct).err()
    }

    fn root_ty_var(&self, var: TyVid) -> TyVid {
        self.root_var(var)
    }

    fn root_const_var(&self, var: ConstVid) -> ConstVid {
        self.root_const_var(var)
    }

    fn opportunistic_resolve_ty_var(&self, vid: TyVid) -> Ty<'db> {
        match self.probe_ty_var(vid) {
            Ok(ty) => ty,
            Err(_) => Ty::new_var(self.interner, self.root_var(vid)),
        }
    }

    fn opportunistic_resolve_int_var(&self, vid: IntVid) -> Ty<'db> {
        self.opportunistic_resolve_int_var(vid)
    }

    fn opportunistic_resolve_float_var(&self, vid: FloatVid) -> Ty<'db> {
        self.opportunistic_resolve_float_var(vid)
    }

    fn opportunistic_resolve_ct_var(&self, vid: ConstVid) -> Const<'db> {
        match self.probe_const_var(vid) {
            Ok(ct) => ct,
            Err(_) => Const::new_var(self.interner, self.root_const_var(vid)),
        }
    }

    fn opportunistic_resolve_lt_var(&self, vid: RegionVid) -> Region<'db> {
        self.inner
            .borrow_mut()
            .unwrap_region_constraints()
            .opportunistic_resolve_var(self.interner, vid)
    }

    fn is_changed_arg(&self, arg: <Self::Interner as rustc_type_ir::Interner>::GenericArg) -> bool {
        match arg.kind() {
            GenericArgKind::Lifetime(_) => {
                // Lifetimes should not change affect trait selection.
                false
            }
            GenericArgKind::Type(ty) => {
                if let TyKind::Infer(infer_ty) = ty.kind() {
                    match infer_ty {
                        InferTy::TyVar(vid) => {
                            !self.probe_ty_var(vid).is_err_and(|_| self.root_var(vid) == vid)
                        }
                        InferTy::IntVar(vid) => {
                            let mut inner = self.inner.borrow_mut();
                            !matches!(
                                inner.int_unification_table().probe_value(vid),
                                IntVarValue::Unknown
                                    if inner.int_unification_table().find(vid) == vid
                            )
                        }
                        InferTy::FloatVar(vid) => {
                            let mut inner = self.inner.borrow_mut();
                            !matches!(
                                inner.float_unification_table().probe_value(vid),
                                FloatVarValue::Unknown
                                    if inner.float_unification_table().find(vid) == vid
                            )
                        }
                        InferTy::FreshTy(_) | InferTy::FreshIntTy(_) | InferTy::FreshFloatTy(_) => {
                            true
                        }
                    }
                } else {
                    true
                }
            }
            GenericArgKind::Const(ct) => {
                if let ConstKind::Infer(infer_ct) = ct.kind() {
                    match infer_ct {
                        InferConst::Var(vid) => !self
                            .probe_const_var(vid)
                            .is_err_and(|_| self.root_const_var(vid) == vid),
                        InferConst::Fresh(_) => true,
                    }
                } else {
                    true
                }
            }
        }
    }

    fn next_ty_infer(&self) -> Ty<'db> {
        self.next_ty_var()
    }

    fn next_region_infer(&self) -> <Self::Interner as rustc_type_ir::Interner>::Region {
        self.next_region_var()
    }

    fn next_const_infer(&self) -> Const<'db> {
        self.next_const_var()
    }

    fn fresh_args_for_item(&self, def_id: SolverDefId) -> GenericArgs<'db> {
        self.fresh_args_for_item(def_id)
    }

    fn instantiate_binder_with_infer<T: TypeFoldable<DbInterner<'db>> + Clone>(
        &self,
        value: Binder<'db, T>,
    ) -> T {
        self.instantiate_binder_with_fresh_vars(BoundRegionConversionTime::HigherRankedType, value)
    }

    fn enter_forall<T: TypeFoldable<DbInterner<'db>> + Clone, U>(
        &self,
        value: Binder<'db, T>,
        f: impl FnOnce(T) -> U,
    ) -> U {
        self.enter_forall(value, f)
    }

    fn equate_ty_vids_raw(&self, a: rustc_type_ir::TyVid, b: rustc_type_ir::TyVid) {
        self.inner.borrow_mut().type_variables().equate(a, b);
    }

    fn equate_int_vids_raw(&self, a: rustc_type_ir::IntVid, b: rustc_type_ir::IntVid) {
        self.inner.borrow_mut().int_unification_table().union(a, b);
    }

    fn equate_float_vids_raw(&self, a: rustc_type_ir::FloatVid, b: rustc_type_ir::FloatVid) {
        self.inner.borrow_mut().float_unification_table().union(a, b);
    }

    fn equate_const_vids_raw(&self, a: rustc_type_ir::ConstVid, b: rustc_type_ir::ConstVid) {
        self.inner.borrow_mut().const_unification_table().union(a, b);
    }

    fn instantiate_ty_var_raw<R: PredicateEmittingRelation<Self>>(
        &self,
        relation: &mut R,
        target_is_expected: bool,
        target_vid: rustc_type_ir::TyVid,
        instantiation_variance: rustc_type_ir::Variance,
        source_ty: Ty<'db>,
    ) -> RelateResult<'db, ()> {
        self.instantiate_ty_var(
            relation,
            target_is_expected,
            target_vid,
            instantiation_variance,
            source_ty,
        )
    }

    fn instantiate_int_var_raw(
        &self,
        vid: rustc_type_ir::IntVid,
        value: rustc_type_ir::IntVarValue,
    ) {
        self.inner.borrow_mut().int_unification_table().union_value(vid, value);
    }

    fn instantiate_float_var_raw(
        &self,
        vid: rustc_type_ir::FloatVid,
        value: rustc_type_ir::FloatVarValue,
    ) {
        self.inner.borrow_mut().float_unification_table().union_value(vid, value);
    }

    fn instantiate_const_var_raw<R: PredicateEmittingRelation<Self>>(
        &self,
        relation: &mut R,
        target_is_expected: bool,
        target_vid: rustc_type_ir::ConstVid,
        source_ct: Const<'db>,
    ) -> RelateResult<'db, ()> {
        self.instantiate_const_var(relation, target_is_expected, target_vid, source_ct)
    }

    fn set_tainted_by_errors(&self, e: ErrorGuaranteed) {
        self.set_tainted_by_errors(e)
    }

    fn shallow_resolve(&self, ty: Ty<'db>) -> Ty<'db> {
        self.shallow_resolve(ty)
    }
    fn shallow_resolve_const(&self, ct: Const<'db>) -> Const<'db> {
        self.shallow_resolve_const(ct)
    }

    fn resolve_vars_if_possible<T>(&self, value: T) -> T
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        self.resolve_vars_if_possible(value)
    }

    fn probe<T>(&self, probe: impl FnOnce() -> T) -> T {
        self.probe(|_| probe())
    }

    fn sub_regions(&self, sub: Region<'db>, sup: Region<'db>, _span: Span) {
        self.inner.borrow_mut().unwrap_region_constraints().make_subregion(sub, sup);
    }

    fn equate_regions(&self, a: Region<'db>, b: Region<'db>, _span: Span) {
        self.inner.borrow_mut().unwrap_region_constraints().make_eqregion(a, b);
    }

    fn register_ty_outlives(&self, _ty: Ty<'db>, _r: Region<'db>, _span: Span) {
        // self.register_type_outlives_constraint(ty, r, &ObligationCause::dummy());
    }

    type OpaqueTypeStorageEntries = OpaqueTypeStorageEntries;

    fn opaque_types_storage_num_entries(&self) -> OpaqueTypeStorageEntries {
        self.inner.borrow_mut().opaque_types().num_entries()
    }
    fn clone_opaque_types_lookup_table(&self) -> Vec<(OpaqueTypeKey<'db>, Ty<'db>)> {
        self.inner.borrow_mut().opaque_types().iter_lookup_table().map(|(k, h)| (k, h.ty)).collect()
    }
    fn clone_duplicate_opaque_types(&self) -> Vec<(OpaqueTypeKey<'db>, Ty<'db>)> {
        self.inner
            .borrow_mut()
            .opaque_types()
            .iter_duplicate_entries()
            .map(|(k, h)| (k, h.ty))
            .collect()
    }
    fn clone_opaque_types_added_since(
        &self,
        prev_entries: OpaqueTypeStorageEntries,
    ) -> Vec<(OpaqueTypeKey<'db>, Ty<'db>)> {
        self.inner
            .borrow_mut()
            .opaque_types()
            .opaque_types_added_since(prev_entries)
            .map(|(k, h)| (k, h.ty))
            .collect()
    }

    fn register_hidden_type_in_storage(
        &self,
        opaque_type_key: OpaqueTypeKey<'db>,
        hidden_ty: Ty<'db>,
        _span: Span,
    ) -> Option<Ty<'db>> {
        self.register_hidden_type_in_storage(opaque_type_key, OpaqueHiddenType { ty: hidden_ty })
    }
    fn add_duplicate_opaque_type(
        &self,
        opaque_type_key: OpaqueTypeKey<'db>,
        hidden_ty: Ty<'db>,
        _span: Span,
    ) {
        self.inner
            .borrow_mut()
            .opaque_types()
            .add_duplicate(opaque_type_key, OpaqueHiddenType { ty: hidden_ty })
    }

    fn reset_opaque_types(&self) {
        let _ = self.take_opaque_types();
    }

    fn sub_unification_table_root_var(&self, var: rustc_type_ir::TyVid) -> rustc_type_ir::TyVid {
        self.sub_unification_table_root_var(var)
    }

    fn sub_unify_ty_vids_raw(&self, a: rustc_type_ir::TyVid, b: rustc_type_ir::TyVid) {
        self.sub_unify_ty_vids_raw(a, b);
    }

    fn opaques_with_sub_unified_hidden_type(
        &self,
        _ty: TyVid,
    ) -> Vec<rustc_type_ir::AliasTy<Self::Interner>> {
        // FIXME: I guess we are okay without this for now since currently r-a lacks of
        // detailed checks over opaque types. Might need to implement this in future.
        vec![]
    }
}
