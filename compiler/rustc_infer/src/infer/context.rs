//! Definition of `InferCtxtLike` from the librarified type layer.
use rustc_data_structures::sso::SsoHashMap;
use rustc_hir::def_id::DefId;
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::relate::RelateResult;
use rustc_middle::ty::relate::combine::PredicateEmittingRelation;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span};
use rustc_type_ir::{TypeSuperFoldable, TypeVisitableExt};

use super::type_variable::TypeVariableValue;
use super::{
    BoundRegionConversionTime, ConstVariableValue, InferCtxt, OpaqueTypeStorageEntries,
    RegionVariableOrigin, SubregionOrigin,
};

impl<'tcx> rustc_type_ir::InferCtxtLike for InferCtxt<'tcx> {
    type Interner = TyCtxt<'tcx>;

    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn next_trait_solver(&self) -> bool {
        self.next_trait_solver
    }

    fn enable_next_solver_overflow_fcw(&self) -> bool {
        self.enable_next_solver_overflow_fcw
    }

    fn disable_trait_solver_fast_paths(&self) -> bool {
        self.disable_trait_solver_fast_paths()
    }

    fn typing_mode_raw(&self) -> ty::TypingMode<'tcx> {
        self.typing_mode_raw()
    }

    fn universe(&self) -> ty::UniverseIndex {
        self.universe()
    }

    fn create_next_universe(&self) -> ty::UniverseIndex {
        self.create_next_universe()
    }

    fn insert_placeholder_assumptions(
        &self,
        u: ty::UniverseIndex,
        assumptions: Option<rustc_type_ir::region_constraint::Assumptions<TyCtxt<'tcx>>>,
    ) {
        self.placeholder_assumptions_for_next_solver.borrow_mut().insert(u, assumptions);
    }

    fn get_placeholder_assumptions(
        &self,
        u: ty::UniverseIndex,
    ) -> Option<rustc_type_ir::region_constraint::Assumptions<TyCtxt<'tcx>>> {
        self.placeholder_assumptions_for_next_solver.borrow().get(&u).unwrap().as_ref().cloned()
    }

    fn get_solver_region_constraint(
        &self,
    ) -> rustc_type_ir::region_constraint::RegionConstraint<TyCtxt<'tcx>> {
        self.inner.borrow().solver_region_constraint_storage.get_constraint()
    }

    fn overwrite_solver_region_constraint(
        &self,
        constraint: rustc_type_ir::region_constraint::RegionConstraint<TyCtxt<'tcx>>,
    ) {
        let mut inner = self.inner.borrow_mut();
        use rustc_data_structures::undo_log::UndoLogs;

        use crate::infer::UndoLog;
        let old_constraint = inner.solver_region_constraint_storage.get_constraint();
        inner.undo_log.push(UndoLog::OverwriteSolverRegionConstraint { old_constraint });
        inner.solver_region_constraint_storage.overwrite_solver_region_constraint(constraint);
    }

    fn universe_of_ty(&self, vid: ty::TyVid) -> Option<ty::UniverseIndex> {
        match self.try_resolve_ty_var(vid) {
            Err(universe) => Some(universe),
            Ok(_) => None,
        }
    }

    fn universe_of_lt(&self, lt: ty::RegionVid) -> Option<ty::UniverseIndex> {
        match self.inner.borrow_mut().unwrap_region_constraints().probe_value(lt) {
            Err(universe) => Some(universe),
            Ok(_) => None,
        }
    }

    fn universe_of_ct(&self, ct: ty::ConstVid) -> Option<ty::UniverseIndex> {
        match self.try_resolve_const_var(ct) {
            Err(universe) => Some(universe),
            Ok(_) => None,
        }
    }

    fn root_ty_var(&self, var: ty::TyVid) -> ty::TyVid {
        self.root_var(var)
    }

    fn sub_unification_table_root_var(&self, var: ty::TyVid) -> ty::TyVid {
        self.sub_unification_table_root_var(var)
    }

    fn root_const_var(&self, var: ty::ConstVid) -> ty::ConstVid {
        self.root_const_var(var)
    }

    fn opportunistic_resolve_ty_var(&self, vid: ty::TyVid) -> Ty<'tcx> {
        match self.try_resolve_ty_var(vid) {
            Ok(ty) => ty,
            Err(_) => Ty::new_var(self.tcx, self.root_var(vid)),
        }
    }

    fn opportunistic_resolve_int_var(&self, vid: ty::IntVid) -> Ty<'tcx> {
        self.opportunistic_resolve_int_var(vid)
    }

    fn opportunistic_resolve_float_var(&self, vid: ty::FloatVid) -> Ty<'tcx> {
        self.opportunistic_resolve_float_var(vid)
    }

    fn opportunistic_resolve_ct_var(&self, vid: ty::ConstVid) -> ty::Const<'tcx> {
        match self.try_resolve_const_var(vid) {
            Ok(ct) => ct,
            Err(_) => ty::Const::new_var(self.tcx, self.root_const_var(vid)),
        }
    }

    fn opportunistic_resolve_lt_var(&self, vid: ty::RegionVid) -> ty::Region<'tcx> {
        self.inner.borrow_mut().unwrap_region_constraints().opportunistic_resolve_var(self.tcx, vid)
    }

    fn is_changed_arg(&self, arg: ty::GenericArg<'tcx>) -> bool {
        match arg.kind() {
            ty::GenericArgKind::Lifetime(_) => {
                // Lifetimes should not change affect trait selection.
                false
            }
            ty::GenericArgKind::Type(ty) => {
                if let ty::Infer(infer_ty) = *ty.kind() {
                    match infer_ty {
                        ty::InferTy::TyVar(vid) => {
                            !self.try_resolve_ty_var(vid).is_err_and(|_| self.root_var(vid) == vid)
                        }
                        ty::InferTy::IntVar(vid) => {
                            let mut inner = self.inner.borrow_mut();
                            !matches!(
                                inner.int_unification_table().probe_value(vid),
                                ty::IntVarValue::Unknown
                                    if inner.int_unification_table().find(vid) == vid
                            )
                        }
                        ty::InferTy::FloatVar(vid) => {
                            let mut inner = self.inner.borrow_mut();
                            !matches!(
                                inner.float_unification_table().probe_value(vid),
                                ty::FloatVarValue::Unknown
                                    if inner.float_unification_table().find(vid) == vid
                            )
                        }
                        ty::InferTy::FreshTy(_)
                        | ty::InferTy::FreshIntTy(_)
                        | ty::InferTy::FreshFloatTy(_) => true,
                    }
                } else {
                    true
                }
            }
            ty::GenericArgKind::Const(ct) => {
                if let ty::ConstKind::Infer(infer_ct) = ct.kind() {
                    match infer_ct {
                        ty::InferConst::Var(vid) => !self
                            .try_resolve_const_var(vid)
                            .is_err_and(|_| self.root_const_var(vid) == vid),
                        ty::InferConst::Fresh(_) => true,
                    }
                } else {
                    true
                }
            }
        }
    }

    fn next_region_infer(&self) -> ty::Region<'tcx> {
        self.next_region_var(RegionVariableOrigin::Misc(DUMMY_SP))
    }

    fn next_ty_infer(&self) -> Ty<'tcx> {
        self.next_ty_var(DUMMY_SP)
    }

    fn next_const_infer(&self) -> ty::Const<'tcx> {
        self.next_const_var(DUMMY_SP)
    }

    fn fresh_args_for_item(&self, def_id: DefId) -> ty::GenericArgsRef<'tcx> {
        self.fresh_args_for_item(DUMMY_SP, def_id)
    }

    fn instantiate_binder_with_infer<T: TypeFoldable<TyCtxt<'tcx>> + Copy>(
        &self,
        value: ty::Binder<'tcx, T>,
    ) -> T {
        self.instantiate_binder_with_fresh_vars(
            DUMMY_SP,
            BoundRegionConversionTime::HigherRankedType,
            value,
        )
    }

    fn enter_forall_without_assumptions<T: TypeFoldable<TyCtxt<'tcx>>, U>(
        &self,
        value: ty::Binder<'tcx, T>,
        f: impl FnOnce(T) -> U,
    ) -> U {
        self.enter_forall(value, f)
    }

    fn enter_forall_with_empty_assumptions<T: TypeFoldable<TyCtxt<'tcx>>, U>(
        &self,
        value: ty::Binder<'tcx, T>,
        f: impl FnOnce(T) -> U,
    ) -> U {
        self.enter_forall(value, |value| {
            let u = self.universe();
            self.placeholder_assumptions_for_next_solver
                .borrow_mut()
                .insert(u, Some(rustc_type_ir::region_constraint::Assumptions::empty()));
            f(value)
        })
    }

    fn equate_ty_vids_raw(&self, a: ty::TyVid, b: ty::TyVid) {
        self.inner.borrow_mut().type_variables().equate(a, b);
    }

    fn sub_unify_ty_vids_raw(&self, a: ty::TyVid, b: ty::TyVid) {
        self.sub_unify_ty_vids_raw(a, b);
    }

    fn equate_int_vids_raw(&self, a: ty::IntVid, b: ty::IntVid) {
        self.inner.borrow_mut().int_unification_table().union(a, b);
    }

    fn equate_float_vids_raw(&self, a: ty::FloatVid, b: ty::FloatVid) {
        self.inner.borrow_mut().float_unification_table().union(a, b);
    }

    fn equate_const_vids_raw(&self, a: ty::ConstVid, b: ty::ConstVid) {
        self.inner.borrow_mut().const_unification_table().union(a, b);
    }

    fn instantiate_ty_var_raw(&self, vid: ty::TyVid, ty: Ty<'tcx>) {
        let ty = lower_universe(self, self.try_resolve_ty_var(vid).unwrap_err(), ty);

        self.inner.borrow_mut().type_variables().instantiate(vid, ty);
    }

    fn instantiate_const_var_raw(&self, vid: ty::ConstVid, ct: ty::Const<'tcx>) {
        let ct = lower_universe(self, self.try_resolve_const_var(vid).unwrap_err(), ct);

        self.inner
            .borrow_mut()
            .const_unification_table()
            .union_value(vid, ConstVariableValue::Known { value: ct });
    }

    fn instantiate_ty_var<R: PredicateEmittingRelation<Self>>(
        &self,
        relation: &mut R,
        target_is_expected: bool,
        target_vid: ty::TyVid,
        instantiation_variance: ty::Variance,
        source_ty: Ty<'tcx>,
    ) -> RelateResult<'tcx, ()> {
        self.instantiate_ty_var(
            relation,
            target_is_expected,
            target_vid,
            instantiation_variance,
            source_ty,
        )
    }

    fn instantiate_int_var_raw(&self, vid: ty::IntVid, value: ty::IntVarValue) {
        self.inner.borrow_mut().int_unification_table().union_value(vid, value);
    }

    fn instantiate_float_var_raw(&self, vid: ty::FloatVid, value: ty::FloatVarValue) {
        self.inner.borrow_mut().float_unification_table().union_value(vid, value);
    }

    fn instantiate_const_var<R: PredicateEmittingRelation<Self>>(
        &self,
        relation: &mut R,
        target_is_expected: bool,
        target_vid: ty::ConstVid,
        source_ct: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ()> {
        self.instantiate_const_var(relation, target_is_expected, target_vid, source_ct)
    }

    fn set_tainted_by_errors(&self, e: ErrorGuaranteed) {
        self.set_tainted_by_errors(e)
    }

    fn shallow_resolve(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.shallow_resolve(ty)
    }
    fn shallow_resolve_const(&self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        self.shallow_resolve_const(ct)
    }

    fn resolve_vars_if_possible<T>(&self, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.resolve_vars_if_possible(value)
    }

    fn probe<T>(&self, probe: impl FnOnce() -> T) -> T {
        self.probe(|_| probe())
    }

    fn commit_if_ok<T, E>(&self, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
        self.commit_if_ok(|_| f())
    }

    fn sub_regions(
        &self,
        sub: ty::Region<'tcx>,
        sup: ty::Region<'tcx>,
        vis: ty::VisibleForLeakCheck,
        span: Span,
    ) {
        self.inner.borrow_mut().unwrap_region_constraints().make_subregion(
            SubregionOrigin::RelateRegionParamBound(span, None),
            sub,
            sup,
            vis,
        );
    }

    fn equate_regions(
        &self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
        vis: ty::VisibleForLeakCheck,
        span: Span,
    ) {
        self.inner.borrow_mut().unwrap_region_constraints().make_eqregion(
            SubregionOrigin::RelateRegionParamBound(span, None),
            a,
            b,
            vis,
        );
    }

    fn register_solver_region_constraint(
        &self,
        c: rustc_type_ir::region_constraint::RegionConstraint<TyCtxt<'tcx>>,
    ) {
        let mut inner = self.inner.borrow_mut();
        use rustc_data_structures::undo_log::UndoLogs;

        use crate::infer::UndoLog;
        inner.undo_log.push(UndoLog::PushSolverRegionConstraint);
        inner.solver_region_constraint_storage.push(c);
    }

    fn register_ty_outlives(&self, ty: Ty<'tcx>, r: ty::Region<'tcx>, span: Span) {
        self.register_type_outlives_constraint(ty, r, &ObligationCause::dummy_with_span(span));
    }

    type OpaqueTypeStorageEntries = OpaqueTypeStorageEntries;
    fn opaque_types_storage_num_entries(&self) -> OpaqueTypeStorageEntries {
        self.inner.borrow_mut().opaque_types().num_entries()
    }
    fn clone_opaque_types_lookup_table(&self) -> Vec<(ty::OpaqueTypeKey<'tcx>, Ty<'tcx>)> {
        self.inner.borrow_mut().opaque_types().iter_lookup_table().map(|(k, h)| (k, h.ty)).collect()
    }
    fn clone_duplicate_opaque_types(&self) -> Vec<(ty::OpaqueTypeKey<'tcx>, Ty<'tcx>)> {
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
    ) -> Vec<(ty::OpaqueTypeKey<'tcx>, Ty<'tcx>)> {
        self.inner
            .borrow_mut()
            .opaque_types()
            .opaque_types_added_since(prev_entries)
            .map(|(k, h)| (k, h.ty))
            .collect()
    }
    fn opaques_with_sub_unified_hidden_type(&self, ty: ty::TyVid) -> Vec<ty::OpaqueAliasTy<'tcx>> {
        self.opaques_with_sub_unified_hidden_type(ty)
    }

    fn register_hidden_type_in_storage(
        &self,
        opaque_type_key: ty::OpaqueTypeKey<'tcx>,
        hidden_ty: Ty<'tcx>,
        span: Span,
    ) -> Option<Ty<'tcx>> {
        self.register_hidden_type_in_storage(
            opaque_type_key,
            ty::ProvisionalHiddenType { span, ty: hidden_ty },
        )
    }
    fn add_duplicate_opaque_type(
        &self,
        opaque_type_key: ty::OpaqueTypeKey<'tcx>,
        hidden_ty: Ty<'tcx>,
        span: Span,
    ) {
        self.inner
            .borrow_mut()
            .opaque_types()
            .add_duplicate(opaque_type_key, ty::ProvisionalHiddenType { span, ty: hidden_ty })
    }

    fn reset_opaque_types(&self) {
        let _ = self.take_opaque_types();
    }
}

fn lower_universe<'tcx, T: TypeFoldable<TyCtxt<'tcx>> + Copy>(
    infcx: &InferCtxt<'tcx>,
    for_universe: ty::UniverseIndex,
    value: T,
) -> T {
    let value = value.fold_with(&mut LowerUniverseFolder {
        infcx,
        for_universe,
        cache: Default::default(),
    });

    // This assertion is needed because we don't lower the universes of placeholders
    // in the folder.
    #[cfg(debug_assertions)]
    {
        let value_universe = ty::max_universe(infcx, value);
        assert!(
            for_universe.can_name(value_universe),
            "variable in universe {:?} can't name value in universe {:?}",
            for_universe,
            value_universe,
        );
    }

    value
}

/// Canonicalizing inputs puts all inference variables and placeholders
/// into the root universe.
///
/// This means when instantiating the query response we need to pull
/// down the universe of returned `var_values` to the universe of
/// the inference variable in `orig_values`.
///
/// This folder is similar to the `Generalizer`, except that it simply
/// structurally folds non-rigid aliases as these should have already
/// been generalized in the query so we shouldn't try to do it again.
struct LowerUniverseFolder<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
    for_universe: ty::UniverseIndex,
    cache: SsoHashMap<Ty<'tcx>, Ty<'tcx>>,
}
impl<'a, 'tcx> ty::TypeFolder<TyCtxt<'tcx>> for LowerUniverseFolder<'a, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if !(t.has_free_regions() || t.has_infer()) {
            return t;
        }

        if let Some(&answer) = self.cache.get(&t) {
            return answer;
        }

        let folded = match t.kind() {
            ty::Infer(ty::TyVar(vid)) => {
                let vid = self.infcx.root_var(*vid);
                let probe = self.infcx.inner.borrow_mut().type_variables().probe(vid);
                match probe {
                    TypeVariableValue::Known { value: u } => u.super_fold_with(self),
                    TypeVariableValue::Unknown { universe } => {
                        if self.for_universe.can_name(universe) {
                            t
                        } else {
                            let mut inner = self.infcx.inner.borrow_mut();
                            let origin = inner.type_variables().var_origin(vid);
                            let new_var_id =
                                inner.type_variables().new_var(self.for_universe, origin);
                            inner.type_variables().equate(vid, new_var_id);
                            Ty::new_var(self.cx(), new_var_id)
                        }
                    }
                }
            }
            _ => t.super_fold_with(self),
        };

        self.cache.insert(t, folded);
        folded
    }

    fn fold_const(&mut self, c: ty::Const<'tcx>) -> ty::Const<'tcx> {
        if !(c.has_free_regions() || c.has_infer()) {
            return c;
        }

        match c.kind() {
            ty::ConstKind::Infer(ty::InferConst::Var(vid)) => {
                let vid = self.infcx.root_const_var(vid);
                let universe = self.infcx.try_resolve_const_var(vid).unwrap_err();
                if self.for_universe.can_name(universe) {
                    c
                } else {
                    let origin = self.infcx.const_var_origin(vid).unwrap();
                    let new_var_id = self
                        .infcx
                        .inner
                        .borrow_mut()
                        .const_unification_table()
                        .new_key(ConstVariableValue::Unknown {
                            origin,
                            universe: self.for_universe,
                        })
                        .vid;

                    self.infcx.inner.borrow_mut().const_unification_table().union(vid, new_var_id);

                    ty::Const::new_var(self.cx(), new_var_id)
                }
            }
            _ => c.super_fold_with(self),
        }
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match r.kind() {
            ty::ReBound(..) | ty::ReErased => r,
            _ => {
                let r_universe = self.infcx.universe_of_region(r);
                if self.for_universe.can_name(r_universe) {
                    r
                } else {
                    // FIXME: unfortunately we lose the relating span here unless we take another
                    // argument.
                    let new_region = self.infcx.next_region_var_in_universe(
                        RegionVariableOrigin::Misc(DUMMY_SP),
                        self.for_universe,
                    );
                    self.infcx.equate_regions(
                        SubregionOrigin::RelateRegionParamBound(DUMMY_SP, None),
                        r,
                        new_region,
                        ty::VisibleForLeakCheck::Yes,
                    );
                    new_region
                }
            }
        }
    }
}
