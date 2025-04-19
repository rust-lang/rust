use std::iter;
use std::rc::Rc;

use rustc_data_structures::frozen::Frozen;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_index::IndexVec;
use rustc_infer::infer::{InferCtxt, NllRegionVariableOrigin};
use rustc_macros::extension;
use rustc_middle::bug;
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::ty::relate::{
    Relate, RelateResult, TypeRelation, structurally_relate_consts, structurally_relate_tys,
};
use rustc_middle::ty::{
    self, DefiningScopeKind, GenericArgsRef, OpaqueHiddenType, OpaqueTypeKey, Region, RegionVid,
    Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeSuperVisitable, TypeVisitable,
    TypeVisitableExt, TypeVisitor, fold_regions,
};
use rustc_mir_dataflow::points::DenseLocationMap;
use rustc_span::{ErrorGuaranteed, Span};
use rustc_trait_selection::error_reporting::infer::region::unexpected_hidden_region_diagnostic;
use rustc_trait_selection::opaque_types::{
    InvalidOpaqueTypeArgs, check_opaque_type_parameter_valid,
};
use tracing::{debug, instrument};

use super::reverse_sccs::ReverseSccGraph;
use super::values::RegionValues;
use super::{ConstraintSccs, RegionDefinition, RegionInferenceContext};
use crate::constraints::ConstraintSccIndex;
use crate::consumers::OutlivesConstraint;
use crate::type_check::free_region_relations::UniversalRegionRelations;
use crate::type_check::{Locations, MirTypeckRegionConstraints};
use crate::universal_regions::{RegionClassification, UniversalRegions};
use crate::{BorrowCheckRootCtxt, BorrowckInferCtxt, BorrowckState};

#[allow(unused)] // TODO
pub(crate) enum DeferredOpaqueTypeError<'tcx> {
    UnexpectedHiddenRegion {
        /// The opaque type.
        opaque_type_key: OpaqueTypeKey<'tcx>,
        /// The hidden type containing the member region.
        hidden_type: OpaqueHiddenType<'tcx>,
        /// The unexpected region.
        member_region: Region<'tcx>,
    },
    InvalidOpaqueTypeArgs(InvalidOpaqueTypeArgs<'tcx>),
}

impl<'tcx> BorrowCheckRootCtxt<'tcx> {
    pub(crate) fn handle_opaque_type_uses(&mut self, borrowck_state: &mut BorrowckState<'tcx>) {
        let BorrowckState {
            infcx,
            constraints,
            universal_region_relations,
            location_map,
            deferred_opaque_type_errors,
            ..
        } = borrowck_state;

        handle_opaque_type_uses(
            self,
            infcx,
            constraints,
            deferred_opaque_type_errors,
            universal_region_relations,
            location_map,
        )
    }
}

struct RegionCtxt<'a, 'tcx> {
    infcx: &'a BorrowckInferCtxt<'tcx>,
    definitions: IndexVec<RegionVid, RegionDefinition<'tcx>>,
    universal_region_relations: &'a UniversalRegionRelations<'tcx>,
    constraint_sccs: ConstraintSccs,
    rev_scc_graph: ReverseSccGraph,
    scc_values: RegionValues<ConstraintSccIndex>,
}

impl<'a, 'tcx> RegionCtxt<'a, 'tcx> {
    fn new(
        infcx: &'a BorrowckInferCtxt<'tcx>,
        universal_region_relations: &'a Frozen<UniversalRegionRelations<'tcx>>,
        location_map: Rc<DenseLocationMap>,
        constraints: &MirTypeckRegionConstraints<'tcx>,
    ) -> RegionCtxt<'a, 'tcx> {
        let mut definitions: IndexVec<_, _> = infcx
            .get_region_var_infos()
            .iter()
            .map(|info| RegionDefinition::new(info.universe, info.origin))
            .collect();

        for (external_name, variable) in
            universal_region_relations.universal_regions.named_universal_regions_iter()
        {
            definitions[variable].external_name = Some(external_name);
        }

        let universal_regions = &universal_region_relations.universal_regions;
        let fr_static = universal_regions.fr_static;
        let constraint_sccs =
            constraints.outlives_constraints.compute_sccs(fr_static, &definitions);
        let rev_scc_graph = ReverseSccGraph::compute(&constraint_sccs, universal_regions);
        // Unlike the `RegionInferenceContext`, we only care about free regions
        // and fully ignore liveness and placeholders.
        let placeholder_indices = Default::default();
        let mut scc_values =
            RegionValues::new(location_map, universal_regions.len(), placeholder_indices);
        for variable in definitions.indices() {
            let scc = constraint_sccs.scc(variable);
            match definitions[variable].origin {
                NllRegionVariableOrigin::FreeRegion => {
                    scc_values.add_element(scc, variable);
                }
                _ => {}
            }
        }
        scc_values.propagate_constraints(&constraint_sccs);

        RegionCtxt {
            infcx,
            definitions,
            universal_region_relations,
            constraint_sccs,
            rev_scc_graph,
            scc_values,
        }
    }

    fn representative(&self, vid: RegionVid) -> RegionVid {
        let scc = self.constraint_sccs.scc(vid);
        self.constraint_sccs.annotation(scc).representative
    }

    fn universal_regions(&self) -> &UniversalRegions<'tcx> {
        &self.universal_region_relations.universal_regions
    }

    fn eval_equal(&self, r1_vid: RegionVid, r2_vid: RegionVid) -> bool {
        let r1 = self.constraint_sccs.scc(r1_vid);
        let r2 = self.constraint_sccs.scc(r2_vid);

        if r1 == r2 {
            return true;
        }

        let universal_outlives = |sub, sup| {
            self.scc_values.universal_regions_outlived_by(sub).all(|r1| {
                self.scc_values
                    .universal_regions_outlived_by(sup)
                    .any(|r2| self.universal_region_relations.outlives(r2, r1))
            })
        };
        universal_outlives(r1, r2) && universal_outlives(r2, r1)
    }
}

pub(crate) fn handle_opaque_type_uses<'tcx>(
    root_cx: &mut BorrowCheckRootCtxt<'tcx>,
    infcx: &BorrowckInferCtxt<'tcx>,
    constraints: &mut MirTypeckRegionConstraints<'tcx>,
    deferred_errors: &mut Vec<DeferredOpaqueTypeError<'tcx>>,
    universal_region_relations: &Frozen<UniversalRegionRelations<'tcx>>,
    location_map: &Rc<DenseLocationMap>,
) {
    let opaque_types = infcx.take_opaque_types();
    if opaque_types.is_empty() {
        return;
    }

    let tcx = infcx.tcx;
    // We need to eagerly map all regions to NLL vars here, as we need to make sure we've
    // introduced nll vars for all used placeholders.
    let mut opaque_types = opaque_types
        .into_iter()
        .map(|entry| {
            fold_regions(tcx, infcx.resolve_vars_if_possible(entry), |r, _| {
                let vid = if let ty::RePlaceholder(placeholder) = r.kind() {
                    constraints.placeholder_region(infcx, placeholder).as_var()
                } else {
                    universal_region_relations.universal_regions.to_region_vid(r)
                };
                Region::new_var(tcx, vid)
            })
        })
        .collect::<Vec<_>>();

    collect_defining_uses(
        root_cx,
        infcx,
        constraints,
        deferred_errors,
        universal_region_relations,
        location_map,
        &mut opaque_types,
    );

    apply_defining_uses(root_cx, infcx, constraints, universal_region_relations, &opaque_types);
}

#[derive(Debug)]
struct DefiningUse<'tcx> {
    opaque_type_key: OpaqueTypeKey<'tcx>,
    arg_regions: Vec<RegionVid>,
    hidden_type: OpaqueHiddenType<'tcx>,
}

fn collect_defining_uses<'tcx>(
    root_cx: &mut BorrowCheckRootCtxt<'tcx>,
    infcx: &BorrowckInferCtxt<'tcx>,
    constraints: &mut MirTypeckRegionConstraints<'tcx>,
    deferred_errors: &mut Vec<DeferredOpaqueTypeError<'tcx>>,
    universal_region_relations: &Frozen<UniversalRegionRelations<'tcx>>,
    location_map: &Rc<DenseLocationMap>,
    opaque_types: &[(OpaqueTypeKey<'tcx>, OpaqueHiddenType<'tcx>)],
) {
    let tcx = infcx.tcx;
    let rcx =
        RegionCtxt::new(infcx, universal_region_relations, Rc::clone(location_map), constraints);

    let mut defining_uses = vec![];
    for &(opaque_type_key, hidden_type) in opaque_types {
        // Check whether the arguments are fully universal.
        //
        // FIXME: We currently treat `Opaque<'a, 'a>` as a defining use and then emit an error
        // as it's not fully universal. We should share this code with `check_opaque_type_parameter_valid`
        // to only consider actual defining uses as defining.

        let is_non_external_free_region = |r| {
            let vid = rcx.representative(r);
            matches!(rcx.definitions[vid].origin, NllRegionVariableOrigin::FreeRegion)
                && !matches!(
                    rcx.universal_regions().region_classification(vid),
                    Some(RegionClassification::External)
                )
        };
        let arg_regions = opaque_type_key
            .iter_captured_args(tcx)
            .filter_map(|(_, arg)| arg.as_region())
            .map(Region::as_var)
            .map(|r| if is_non_external_free_region(r) { Some(r) } else { None });
        let arg_regions =
            iter::once(Some(rcx.universal_regions().fr_static)).chain(arg_regions).collect();

        if let Some(arg_regions) = arg_regions {
            defining_uses.push(DefiningUse { opaque_type_key, arg_regions, hidden_type });
        }
    }

    debug!(?defining_uses);

    apply_member_constraints(&rcx, constraints, &defining_uses);

    // Applying member constraints may add new region constraints, so we
    // need to recompute the region context before actually mapping the defining
    // uses to the definition site.
    //
    // FIXME: We should be able to partially reuse and optimize this.
    let rcx =
        RegionCtxt::new(infcx, universal_region_relations, Rc::clone(location_map), constraints);
    map_defining_uses_to_definition_site(root_cx, &rcx, &defining_uses, deferred_errors);
}

fn apply_member_constraints<'tcx>(
    rcx: &RegionCtxt<'_, 'tcx>,
    constraints: &mut MirTypeckRegionConstraints<'tcx>,
    defining_uses: &[DefiningUse<'tcx>],
) {
    for DefiningUse { opaque_type_key: _, arg_regions, hidden_type } in defining_uses {
        let mut visitor =
            ApplyMemberConstraintsVisitor { rcx, arg_regions, span: hidden_type.span, constraints };
        hidden_type.ty.visit_with(&mut visitor);
    }
}

struct ApplyMemberConstraintsVisitor<'a, 'tcx> {
    rcx: &'a RegionCtxt<'a, 'tcx>,
    arg_regions: &'a [RegionVid],
    span: Span,
    constraints: &'a mut MirTypeckRegionConstraints<'tcx>,
}
impl<'tcx> ApplyMemberConstraintsVisitor<'_, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.rcx.infcx.tcx
    }
    fn visit_closure_args(&mut self, def_id: DefId, args: GenericArgsRef<'tcx>) {
        let generics = self.cx().generics_of(def_id);
        for arg in args.iter().skip(generics.parent_count) {
            arg.visit_with(self);
        }
    }
}
impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for ApplyMemberConstraintsVisitor<'_, 'tcx> {
    // Invoked for each member region in the hidden type. If possible, we try to lift
    // it to one of the `arg_regions` from the `opaque_type_key`.
    //
    // If there's a unique minimum choice, we emit a `'member: 'min_choice` constraint.
    // Note that we do not require the two regions to be equal... TODO examples
    fn visit_region(&mut self, r: Region<'tcx>) {
        let member_vid = match r.kind() {
            ty::ReBound(..) => return,
            ty::ReVar(vid) => vid,
            _ => unreachable!(),
        };
        let member = self.rcx.constraint_sccs.scc(member_vid);

        // If the member region lives in a higher universe, we currently choose
        // the most conservative option by leaving it unchanged.
        if !self.rcx.constraint_sccs.annotation(member).min_universe().is_root() {
            debug!("member not in root universe");
            return;
        }

        // The existing value of `'member` is a lower-bound. If its is already larger than
        // some universal region, we cannot equate it with that region. Said differently, we
        // ignore choice regions which are smaller than this member region.
        let mut choice_regions = self
            .arg_regions
            .iter()
            .copied()
            .map(|r| self.rcx.representative(r))
            .filter(|&choice_region| {
                self.rcx.scc_values.universal_regions_outlived_by(member).all(|lower_bound| {
                    self.rcx.universal_region_relations.outlives(choice_region, lower_bound)
                })
            })
            .collect::<Vec<_>>();
        debug!(?choice_regions, "after enforcing lower-bound");

        // Now find all the *upper bounds* -- that is, each UB is a
        // free region that must outlive the member region `R0` (`UB:
        // R0`). Therefore, we need only keep an option `O` if `UB: O`
        // for all UB.
        //
        // If we have a requirement `'upper_bound: 'member`, equating `'member`
        // with some region `'choice` means we now also require `'upper_bound: 'choice`.
        // Avoid choice regions for which this does not hold.
        for ub in self.rcx.rev_scc_graph.upper_bounds(member) {
            choice_regions.retain(|&choice_region| {
                self.rcx.universal_region_relations.outlives(ub, choice_region)
            });
        }
        debug!(?choice_regions, "after enforcing upper-bound");

        // At this point we can pick any member of `choice_regions` and would like to choose
        // it to be a small as possible. To avoid potential non-determinism we will pick the
        // smallest such choice.
        //
        // Because universal regions are only partially ordered (i.e, not every two regions are
        // comparable), we will ignore any region that doesn't compare to all others when picking
        // the minimum choice.
        //
        // For example, consider `choice_regions = ['static, 'a, 'b, 'c, 'd, 'e]`, where
        // `'static: 'a, 'static: 'b, 'a: 'c, 'b: 'c, 'c: 'd, 'c: 'e`.
        // `['d, 'e]` are ignored because they do not compare - the same goes for `['a, 'b]`.
        let totally_ordered_subset = choice_regions.iter().copied().filter(|&r1| {
            choice_regions.iter().all(|&r2| {
                self.rcx.universal_region_relations.outlives(r1, r2)
                    || self.rcx.universal_region_relations.outlives(r2, r1)
            })
        });
        // Now we're left with `['static, 'c]`. Pick `'c` as the minimum!
        let Some(min_choice) = totally_ordered_subset.reduce(|r1, r2| {
            let r1_outlives_r2 = self.rcx.universal_region_relations.outlives(r1, r2);
            let r2_outlives_r1 = self.rcx.universal_region_relations.outlives(r2, r1);
            match (r1_outlives_r2, r2_outlives_r1) {
                (true, true) => r1.min(r2),
                (true, false) => r2,
                (false, true) => r1,
                (false, false) => bug!("incomparable regions in total order"),
            }
        }) else {
            debug!("no unique minimum choice");
            return;
        };
        debug!(?min_choice);
        let locations = Locations::All(self.span);
        self.constraints.outlives_constraints.push(OutlivesConstraint {
            sup: member_vid,
            sub: min_choice,
            locations,
            span: self.span,
            category: ConstraintCategory::OpaqueType,
            variance_info: ty::VarianceDiagInfo::None,
            from_closure: false,
        });
    }

    fn visit_ty(&mut self, ty: Ty<'tcx>) {
        if !ty.flags().intersects(ty::TypeFlags::HAS_FREE_REGIONS) {
            return;
        }

        match *ty.kind() {
            ty::Closure(def_id, args)
            | ty::CoroutineClosure(def_id, args)
            | ty::Coroutine(def_id, args) => self.visit_closure_args(def_id, args),

            ty::Alias(kind, ty::AliasTy { def_id, args, .. })
                if let Some(variances) = self.cx().opt_alias_variances(kind, def_id) =>
            {
                // Skip lifetime parameters that are not captured, since they do
                // not need member constraints registered for them; we'll erase
                // them (and hopefully in the future replace them with placeholders).
                for (&v, arg) in std::iter::zip(variances, args.iter()) {
                    if v != ty::Bivariant {
                        arg.visit_with(self)
                    }
                }
            }

            _ => ty.super_visit_with(self),
        }
    }
}

fn map_defining_uses_to_definition_site<'tcx>(
    root_cx: &mut BorrowCheckRootCtxt<'tcx>,
    rcx: &RegionCtxt<'_, 'tcx>,
    defining_uses: &[DefiningUse<'tcx>],
    deferred_errors: &mut Vec<DeferredOpaqueTypeError<'tcx>>,
) {
    for &DefiningUse { opaque_type_key, ref arg_regions, hidden_type } in defining_uses {
        let opaque_type_key = opaque_type_key.fold_captured_lifetime_args(rcx.infcx.tcx, |r| {
            let vid = rcx.representative(r.as_var());
            rcx.definitions[vid].external_name.unwrap()
        });
        // TODO: explain what's going on here
        let mut visitor =
            HiddenTypeMeh { rcx, arg_regions, opaque_type_key, hidden_type, deferred_errors };
        let hidden_type = hidden_type.fold_with(&mut visitor);
        let ty = rcx
            .infcx
            .infer_opaque_definition_from_instantiation(opaque_type_key, hidden_type)
            .unwrap_or_else(|err| {
                deferred_errors.push(DeferredOpaqueTypeError::InvalidOpaqueTypeArgs(err));
                Ty::new_error_with_message(
                    rcx.infcx.tcx,
                    hidden_type.span,
                    "deferred invalid opaque type args",
                )
            });
        root_cx.add_concrete_opaque_type(
            opaque_type_key.def_id,
            OpaqueHiddenType { span: hidden_type.span, ty },
        );
    }
}
struct HiddenTypeMeh<'a, 'tcx> {
    rcx: &'a RegionCtxt<'a, 'tcx>,
    arg_regions: &'a [RegionVid],

    opaque_type_key: OpaqueTypeKey<'tcx>,
    hidden_type: OpaqueHiddenType<'tcx>,
    deferred_errors: &'a mut Vec<DeferredOpaqueTypeError<'tcx>>,
}

impl<'tcx> HiddenTypeMeh<'_, 'tcx> {
    fn fold_closure_args(
        &mut self,
        def_id: DefId,
        args: GenericArgsRef<'tcx>,
    ) -> GenericArgsRef<'tcx> {
        let generics = self.cx().generics_of(def_id);
        self.cx().mk_args_from_iter(args.iter().enumerate().map(|(index, arg)| {
            if index < generics.parent_count {
                self.cx().erase_regions(arg)
            } else {
                arg.fold_with(self)
            }
        }))
    }
}
impl<'tcx> TypeFolder<TyCtxt<'tcx>> for HiddenTypeMeh<'_, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.rcx.infcx.tcx
    }

    fn fold_region(&mut self, r: Region<'tcx>) -> Region<'tcx> {
        match r.kind() {
            // ignore bound regions, keep visiting
            ty::ReBound(_, _) => r,
            _ => self
                .arg_regions
                .iter()
                .copied()
                .find(|&arg_vid| self.rcx.eval_equal(r.as_var(), arg_vid))
                .map(|r| {
                    let vid = self.rcx.representative(r);
                    self.rcx.definitions[vid].external_name.unwrap()
                })
                .unwrap_or_else(|| {
                    self.deferred_errors.push(DeferredOpaqueTypeError::UnexpectedHiddenRegion {
                        hidden_type: self.hidden_type,
                        opaque_type_key: self.opaque_type_key,
                        member_region: r,
                    });
                    ty::Region::new_error_with_message(
                        self.cx(),
                        self.hidden_type.span,
                        "opaque type with non-universal region args",
                    )
                }),
        }
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if !ty.flags().intersects(ty::TypeFlags::HAS_FREE_REGIONS) {
            return ty;
        }

        let tcx = self.cx();
        match *ty.kind() {
            ty::Closure(def_id, args) => {
                Ty::new_closure(tcx, def_id, self.fold_closure_args(def_id, args))
            }

            ty::CoroutineClosure(def_id, args) => {
                Ty::new_coroutine_closure(tcx, def_id, self.fold_closure_args(def_id, args))
            }

            ty::Coroutine(def_id, args) => {
                Ty::new_coroutine(tcx, def_id, self.fold_closure_args(def_id, args))
            }

            ty::Alias(kind, ty::AliasTy { def_id, args, .. })
                if let Some(variances) = tcx.opt_alias_variances(kind, def_id) =>
            {
                // Skip lifetime parameters that are not captured, since they do
                // not need member constraints registered for them; we'll erase
                // them (and hopefully in the future replace them with placeholders).
                let args =
                    tcx.mk_args_from_iter(std::iter::zip(variances, args.iter()).map(|(&v, s)| {
                        if v == ty::Bivariant { tcx.erase_regions(s) } else { s.fold_with(self) }
                    }));
                ty::AliasTy::new_from_args(tcx, def_id, args).to_ty(tcx)
            }

            _ => ty.super_fold_with(self),
        }
    }
}

fn apply_defining_uses<'tcx>(
    root_cx: &mut BorrowCheckRootCtxt<'tcx>,
    infcx: &BorrowckInferCtxt<'tcx>,
    constraints: &mut MirTypeckRegionConstraints<'tcx>,
    universal_region_relations: &Frozen<UniversalRegionRelations<'tcx>>,
    opaque_types: &[(OpaqueTypeKey<'tcx>, OpaqueHiddenType<'tcx>)],
) {
    let tcx = infcx.tcx;
    for &(key, hidden_type) in opaque_types {
        let Some(expected) = root_cx.get_concrete_opaque_type(key.def_id) else {
            let guar =
                tcx.dcx().span_err(hidden_type.span, "non-defining use in the defining scope");
            root_cx.add_concrete_opaque_type(key.def_id, OpaqueHiddenType::new_error(tcx, guar));
            infcx.set_tainted_by_errors(guar);
            continue;
        };

        let expected = ty::fold_regions(tcx, expected.instantiate(tcx, key.args), |re, _dbi| {
            match re.kind() {
                ty::ReErased => infcx.next_nll_region_var(
                    NllRegionVariableOrigin::Existential { from_forall: false },
                    || crate::RegionCtxt::Existential(None),
                ),
                _ => re,
            }
        });

        let universal_regions = &universal_region_relations.universal_regions;
        let mut relation =
            EquateRegions { infcx, span: hidden_type.span, universal_regions, constraints };
        match TypeRelation::relate(&mut relation, hidden_type.ty, expected.ty) {
            Ok(_) => {}
            Err(_) => {
                let _ = hidden_type.build_mismatch_error(&expected, tcx).map(|d| d.emit());
            }
        }

        // normalize -> equate
        //
        // Do we need to normalize?
        // Only to support non-universal type or const args. It feels likely that we need (and want) to do so
        // This once again needs to be careful about cycles: normalizing and equating while defining/revealing
        // opaque types may end up introducing new defining uses.

        // How can we equate here?
        // If we need to normalize the answer is just "it's TypeChecker time"
        // without it we could manually walk over the types using a type relation and equate region vars
        // that way.
    }
}

fn to_region_vid<'tcx>(
    constraints: &MirTypeckRegionConstraints<'tcx>,
    universal_regions: &UniversalRegions<'tcx>,
    r: Region<'tcx>,
) -> RegionVid {
    if let ty::RePlaceholder(placeholder) = r.kind() {
        constraints.get_placeholder_region(placeholder).as_var()
    } else {
        universal_regions.to_region_vid(r)
    }
}

// FUCKING SHIT NO!wegrdtfhergfsg
struct EquateRegions<'a, 'tcx> {
    infcx: &'a BorrowckInferCtxt<'tcx>,
    span: Span,
    universal_regions: &'a UniversalRegions<'tcx>,
    constraints: &'a mut MirTypeckRegionConstraints<'tcx>,
}

impl<'tcx> TypeRelation<TyCtxt<'tcx>> for EquateRegions<'_, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn relate_with_variance<T: Relate<TyCtxt<'tcx>>>(
        &mut self,
        _variance: ty::Variance,
        _info: ty::VarianceDiagInfo<TyCtxt<'tcx>>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        self.relate(a, b)
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        structurally_relate_tys(self, a, b)
    }

    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        if matches!(a.kind(), ty::ReBound(..)) || matches!(b.kind(), ty::ReBound(..)) {
            assert_eq!(a, b);
            return Ok(a);
        }

        let a_vid = to_region_vid(self.constraints, self.universal_regions, a);
        let b_vid = to_region_vid(self.constraints, self.universal_regions, b);
        let locations = Locations::All(self.span);
        self.constraints.outlives_constraints.push(OutlivesConstraint {
            sup: a_vid,
            sub: b_vid,
            locations,
            span: self.span,
            category: ConstraintCategory::OpaqueType,
            variance_info: ty::VarianceDiagInfo::None,
            from_closure: false,
        });
        self.constraints.outlives_constraints.push(OutlivesConstraint {
            sup: b_vid,
            sub: a_vid,
            locations,
            span: self.span,
            category: ConstraintCategory::OpaqueType,
            variance_info: ty::VarianceDiagInfo::None,
            from_closure: false,
        });

        Ok(a)
    }

    fn consts(
        &mut self,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        structurally_relate_consts(self, a, b)
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        b: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<TyCtxt<'tcx>>,
    {
        self.relate(a.skip_binder(), b.skip_binder())?;
        Ok(a)
    }
}

#[extension(pub trait InferCtxtExt<'tcx>)]
impl<'tcx> InferCtxt<'tcx> {
    /// Given the fully resolved, instantiated type for an opaque
    /// type, i.e., the value of an inference variable like C1 or C2
    /// (*), computes the "definition type" for an opaque type
    /// definition -- that is, the inferred value of `Foo1<'x>` or
    /// `Foo2<'x>` that we would conceptually use in its definition:
    /// ```ignore (illustrative)
    /// type Foo1<'x> = impl Bar<'x> = AAA;  // <-- this type AAA
    /// type Foo2<'x> = impl Bar<'x> = BBB;  // <-- or this type BBB
    /// fn foo<'a, 'b>(..) -> (Foo1<'a>, Foo2<'b>) { .. }
    /// ```
    /// Note that these values are defined in terms of a distinct set of
    /// generic parameters (`'x` instead of `'a`) from C1 or C2. The main
    /// purpose of this function is to do that translation.
    ///
    /// (*) C1 and C2 were introduced in the comments on
    /// `register_member_constraints`. Read that comment for more context.
    ///
    /// # Parameters
    ///
    /// - `def_id`, the `impl Trait` type
    /// - `args`, the args used to instantiate this opaque type
    /// - `instantiated_ty`, the inferred type C1 -- fully resolved, lifted version of
    ///   `opaque_defn.concrete_ty`
    #[instrument(level = "debug", skip(self))]
    fn infer_opaque_definition_from_instantiation(
        &self,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        instantiated_ty: OpaqueHiddenType<'tcx>,
    ) -> Result<Ty<'tcx>, InvalidOpaqueTypeArgs<'tcx>> {
        if let Some(guar) = self.tainted_by_errors() {
            return Err(guar.into());
        }
        check_opaque_type_parameter_valid(
            self,
            opaque_type_key,
            instantiated_ty.span,
            DefiningScopeKind::MirBorrowck,
        )?;

        let definition_ty = instantiated_ty
            .remap_generic_params_to_declaration_params(
                opaque_type_key,
                self.tcx,
                DefiningScopeKind::MirBorrowck,
            )
            .ty;

        definition_ty.error_reported()?;
        Ok(definition_ty)
    }
}

impl<'tcx> RegionInferenceContext<'tcx> {
    pub(crate) fn emit_deferred_opaque_type_errors(
        &self,
        root_cx: &mut BorrowCheckRootCtxt<'tcx>,
        infcx: &BorrowckInferCtxt<'tcx>,
        errors: Vec<DeferredOpaqueTypeError<'tcx>>,
    ) {
        let mut prev_hidden_region_errors = FxHashMap::default();
        let mut guar = None;
        for error in errors {
            guar = Some(match error {
                DeferredOpaqueTypeError::UnexpectedHiddenRegion {
                    opaque_type_key,
                    hidden_type,
                    member_region,
                } => self.report_unexpected_hidden_region_errors(
                    root_cx,
                    infcx,
                    &mut prev_hidden_region_errors,
                    opaque_type_key,
                    hidden_type,
                    member_region,
                ),
                DeferredOpaqueTypeError::InvalidOpaqueTypeArgs(err) => err.report(infcx),
            });
        }
        let guar = guar.unwrap();
        root_cx.set_tainted_by_errors(guar);
        infcx.set_tainted_by_errors(guar);
    }

    fn report_unexpected_hidden_region_errors(
        &self,
        root_cx: &mut BorrowCheckRootCtxt<'tcx>,
        infcx: &BorrowckInferCtxt<'tcx>,
        prev_errors: &mut FxHashMap<(Span, Ty<'tcx>, OpaqueTypeKey<'tcx>), ErrorGuaranteed>,
        opaque_type_key: OpaqueTypeKey<'tcx>,
        hidden_type: OpaqueHiddenType<'tcx>,
        member_region: Region<'tcx>,
    ) -> ErrorGuaranteed {
        let tcx = infcx.tcx;
        let named_ty = self.name_regions_for_member_constraint(tcx, hidden_type.ty);
        let named_key = self.name_regions_for_member_constraint(tcx, opaque_type_key);
        let named_region = self.name_regions_for_member_constraint(tcx, member_region);

        *prev_errors.entry((hidden_type.span, named_ty, named_key)).or_insert_with(|| {
            let guar = unexpected_hidden_region_diagnostic(
                infcx,
                root_cx.root_def_id(),
                hidden_type.span,
                named_ty,
                named_region,
                named_key,
            )
            .emit();
            root_cx
                .add_concrete_opaque_type(named_key.def_id, OpaqueHiddenType::new_error(tcx, guar));
            guar
        })
    }

    /// Map the regions in the type to named regions. This is similar to what
    /// `infer_opaque_types` does, but can infer any universal region, not only
    /// ones from the args for the opaque type. It also doesn't double check
    /// that the regions produced are in fact equal to the named region they are
    /// replaced with. This is fine because this function is only to improve the
    /// region names in error messages.
    ///
    /// This differs from `MirBorrowckCtxt::name_regions` since it is particularly
    /// lax with mapping region vids that are *shorter* than a universal region to
    /// that universal region. This is useful for member region constraints since
    /// we want to suggest a universal region name to capture even if it's technically
    /// not equal to the error region.
    fn name_regions_for_member_constraint<T>(&self, tcx: TyCtxt<'tcx>, ty: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        fold_regions(tcx, ty, |region, _| match region.kind() {
            ty::ReVar(vid) => {
                let scc = self.constraint_sccs.scc(vid);

                // Special handling of higher-ranked regions.
                if !self.constraint_sccs.annotation(scc).min_universe().is_root() {
                    match self.scc_values.placeholders_contained_in(scc).enumerate().last() {
                        // If the region contains a single placeholder then they're equal.
                        Some((0, placeholder)) => {
                            return ty::Region::new_placeholder(tcx, placeholder);
                        }

                        // Fallback: this will produce a cryptic error message.
                        _ => return region,
                    }
                }

                // Find something that we can name
                let upper_bound = self.approx_universal_upper_bound(vid);
                if let Some(universal_region) = self.definitions[upper_bound].external_name {
                    return universal_region;
                }

                // Nothing exact found, so we pick a named upper bound, if there's only one.
                // If there's >1 universal region, then we probably are dealing w/ an intersection
                // region which cannot be mapped back to a universal.
                // FIXME: We could probably compute the LUB if there is one.
                let scc = self.constraint_sccs.scc(vid);
                let rev_scc_graph = &ReverseSccGraph::compute(
                    &self.constraint_sccs,
                    &self.universal_region_relations.universal_regions,
                );
                let upper_bounds: Vec<_> = rev_scc_graph
                    .upper_bounds(scc)
                    .filter_map(|vid| self.definitions[vid].external_name)
                    .filter(|r| !r.is_static())
                    .collect();
                match &upper_bounds[..] {
                    [universal_region] => *universal_region,
                    _ => region,
                }
            }
            _ => region,
        })
    }
}
