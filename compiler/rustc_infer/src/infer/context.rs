///! Definition of `InferCtxtLike` from the librarified type layer.
use rustc_hir::def_id::DefId;
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::relate::RelateResult;
use rustc_middle::ty::relate::combine::PredicateEmittingRelation;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span};

use super::{BoundRegionConversionTime, InferCtxt, RegionVariableOrigin, SubregionOrigin};

impl<'tcx> rustc_type_ir::InferCtxtLike for InferCtxt<'tcx> {
    type Interner = TyCtxt<'tcx>;

    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn next_trait_solver(&self) -> bool {
        self.next_trait_solver
    }

    fn typing_mode(&self) -> ty::TypingMode<'tcx> {
        self.typing_mode()
    }

    fn universe(&self) -> ty::UniverseIndex {
        self.universe()
    }

    fn create_next_universe(&self) -> ty::UniverseIndex {
        self.create_next_universe()
    }

    fn universe_of_ty(&self, vid: ty::TyVid) -> Option<ty::UniverseIndex> {
        match self.probe_ty_var(vid) {
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
        match self.probe_const_var(ct) {
            Err(universe) => Some(universe),
            Ok(_) => None,
        }
    }

    fn root_ty_var(&self, var: ty::TyVid) -> ty::TyVid {
        self.root_var(var)
    }

    fn root_const_var(&self, var: ty::ConstVid) -> ty::ConstVid {
        self.root_const_var(var)
    }

    fn opportunistic_resolve_ty_var(&self, vid: ty::TyVid) -> Ty<'tcx> {
        match self.probe_ty_var(vid) {
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
        match self.probe_const_var(vid) {
            Ok(ct) => ct,
            Err(_) => ty::Const::new_var(self.tcx, self.root_const_var(vid)),
        }
    }

    fn opportunistic_resolve_lt_var(&self, vid: ty::RegionVid) -> ty::Region<'tcx> {
        self.inner.borrow_mut().unwrap_region_constraints().opportunistic_resolve_var(self.tcx, vid)
    }

    fn next_region_infer(&self) -> ty::Region<'tcx> {
        self.next_region_var(RegionVariableOrigin::MiscVariable(DUMMY_SP))
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

    fn enter_forall<T: TypeFoldable<TyCtxt<'tcx>>, U>(
        &self,
        value: ty::Binder<'tcx, T>,
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

    fn sub_regions(&self, sub: ty::Region<'tcx>, sup: ty::Region<'tcx>, span: Span) {
        self.inner.borrow_mut().unwrap_region_constraints().make_subregion(
            SubregionOrigin::RelateRegionParamBound(span, None),
            sub,
            sup,
        );
    }

    fn equate_regions(&self, a: ty::Region<'tcx>, b: ty::Region<'tcx>, span: Span) {
        self.inner.borrow_mut().unwrap_region_constraints().make_eqregion(
            SubregionOrigin::RelateRegionParamBound(span, None),
            a,
            b,
        );
    }

    fn register_ty_outlives(&self, ty: Ty<'tcx>, r: ty::Region<'tcx>, span: Span) {
        self.register_region_obligation_with_cause(ty, r, &ObligationCause::dummy_with_span(span));
    }
}
