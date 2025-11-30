// FIXME(fmease): Add somewhat elaborate top-level docs (goals, non-goals, algo, ..).

// NOTE: This algorithm is *incomplete* and will likely remain that way forever.
// * FIXME(#146570): Since we evaluate the trait ref with identity substitutions instead of
//   fresh infer vars, we won't detect cases where there are applicable auto trait candidates
//   for multiple sets of substitutions. Ideally, we would synthesize an impl per generic arg
//   substitutions. Of course, then everything needs to happen in a probe.
// * FIXME(const_trait_impl): Since we currently intentionally refrain from rendering the
//   constness of trait bounds (to avoid leaking the experimental un-RFC'ed into std's docs),
//   we don't bother collecting host-effect predicates.
// * FIXME(fmease): We drop a bunch of predicates that are indirectly "connected to"
//   projection predicates
// * [a whole slew of other cases most likely]

use std::ops::ControlFlow;

use rustc_data_structures::fx::{FxIndexMap, FxIndexSet, IndexEntry};
use rustc_data_structures::thin_vec::ThinVec;
use rustc_data_structures::unord::UnordMap;
use rustc_hir as hir;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::region_constraints::{ConstraintKind, RegionConstraintData};
use rustc_middle::traits::solve::Goal;
use rustc_middle::ty::{
    self, Region, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt, Upcast,
};
use rustc_span::def_id::DefId;
use rustc_span::symbol::Symbol;
use rustc_trait_selection::regions::OutlivesEnvironmentBuildExt;
use rustc_trait_selection::solve::inspect::{InferCtxtProofTreeExt, InspectGoal, ProofTreeVisitor};
use tracing::{debug, instrument};

use crate::clean::{
    self, Lifetime, clean_generic_param_def, clean_middle_ty, clean_predicate,
    clean_trait_ref_with_constraints, clean_ty_generics_inner, simplify,
};
use crate::core::DocContext;

#[instrument(level = "debug", skip(cx))]
pub(crate) fn synthesize_auto_trait_impls<'tcx>(
    cx: &mut DocContext<'tcx>,
    item_def_id: DefId,
) -> Vec<clean::Item> {
    let tcx = cx.tcx;
    let typing_env = ty::TypingEnv::non_body_analysis(tcx, item_def_id);
    let ty = tcx.type_of(item_def_id).instantiate_identity();

    let mut auto_trait_impls: Vec<_> = cx
        .auto_traits
        .clone()
        .into_iter()
        .filter_map(|trait_def_id| {
            synthesize_auto_trait_impl(cx, ty, trait_def_id, typing_env, item_def_id, Mode::Auto)
        })
        .collect();

    // While `Sized` is not an auto trait, our algorithm works just as well for it.
    // Contrary to actual auto traits, we're only interested in the case where the
    // type *doesn't* implemented the trait since most types are `Sized`.
    if !ty.is_sized(tcx, typing_env)
        && let Some(sized_trait_def_id) = tcx.lang_items().sized_trait()
        && let Some(impl_item) = synthesize_auto_trait_impl(
            cx,
            ty,
            sized_trait_def_id,
            typing_env,
            item_def_id,
            Mode::Sized,
        )
    {
        auto_trait_impls.push(impl_item);
    }

    auto_trait_impls
}

#[instrument(level = "debug", skip(cx, ty, typing_env, mode))]
fn synthesize_auto_trait_impl<'tcx>(
    cx: &mut DocContext<'tcx>,
    ty: Ty<'tcx>,
    trait_def_id: DefId,
    typing_env: ty::TypingEnv<'tcx>,
    item_def_id: DefId,
    mode: Mode,
) -> Option<clean::Item> {
    if !cx.generated_synthetics.insert((ty, trait_def_id)) {
        debug!("already generated, aborting");
        return None;
    }

    let tcx = cx.tcx;

    // The mere existence of a user-written impl for that ADT suppresses any potential auto trait
    // impls. This corresponds to auto trait candidate disqualification in the trait solver.
    if let Mode::Auto = mode {
        let mut seen_user_written_impl = false;
        tcx.for_each_relevant_impl(trait_def_id, ty, |_| {
            seen_user_written_impl = true;
        });
        if seen_user_written_impl {
            return None;
        }
    }

    let (infcx, param_env) =
        tcx.infer_ctxt().with_next_trait_solver(true).build_with_typing_env(typing_env);

    let trait_ref = ty::TraitRef::new(tcx, trait_def_id, [ty]);
    let goal = Goal::new(tcx, param_env, trait_ref);

    let mut collector =
        PredicateCollector { clauses: Vec::new(), const_var_tys: UnordMap::default() };

    let (generics, polarity) = match infcx.visit_proof_tree(goal, &mut collector) {
        // The type implements the auto trait.
        ControlFlow::Continue(()) => {
            if let Mode::Sized = mode {
                return None;
            }

            collector.clauses.extend_from_slice(param_env.caller_bounds());
            let param_env = ty::ParamEnv::new(tcx.mk_clauses(&collector.clauses));

            let outlives_env =
                OutlivesEnvironment::new(&infcx, hir::def_id::CRATE_DEF_ID, param_env, []);
            let _ = infcx.process_registered_region_obligations(&outlives_env, |ty, _| Ok(ty));
            let region_data = infcx.inner.borrow_mut().unwrap_region_constraints().data().clone();
            let vid_to_region = map_vid_to_region(&region_data);

            let generics = clean_param_env(
                cx,
                item_def_id,
                ImplInfo {
                    param_env,
                    const_var_tys: collector.const_var_tys,
                    region_data,
                    vid_to_region,
                },
            );

            (generics, ty::ImplPolarity::Positive)
        }
        // The type doesn't implement the auto trait (unconditionally).
        ControlFlow::Break(()) => {
            // FIXME(#146571): We're using negative impls to represent the fact that a type
            // doesn't implement an auto trait. This is technically speaking incorrect since
            // negative impls provide stronger guarantees wrt. coherence and API evolution.
            // Well, it's correct for `Sized` (not an auto trait) because it's fundamental.

            // We neither forward the predicates we've collected, nor the ones from the ParamEnv
            // of the ADT (*) since the "unimplemented-ness" is unconditional.
            let mut generics = clean_ty_generics_inner(
                cx,
                tcx.generics_of(item_def_id),
                ty::GenericPredicates::default(),
            );
            // FIXME: (*) We remove `TyParam: ?Sized` bounds again because someone must've
            //        thought it "looked nicer". However, strictly speaking that's incorrect for
            //        items whose type params are *actually* `?Sized` because then the synthetic
            //        negative impl would be *conditional* which is illegal for auto traits (E0367).
            //
            //        We should either stop clearing relaxed bounds (these impls needn't satisfy the
            //        predicates of the ADT) or just copy over the predicates from the ADT
            //        (see also this tangentially related issue: #111101).
            generics.where_predicates.clear();

            (generics, ty::ImplPolarity::Negative)
        }
    };

    Some(clean::Item {
        inner: Box::new(clean::ItemInner {
            name: None,
            attrs: Default::default(),
            stability: None,
            kind: clean::ImplItem(Box::new(clean::Impl {
                safety: hir::Safety::Safe,
                generics,
                trait_: Some(clean_trait_ref_with_constraints(
                    cx,
                    ty::Binder::dummy(trait_ref),
                    ThinVec::new(),
                )),
                for_: clean_middle_ty(ty::Binder::dummy(ty), cx, None, None),
                items: Vec::new(),
                polarity,
                kind: clean::ImplKind::Auto,
            })),
            item_id: clean::ItemId::Auto { trait_: trait_def_id, for_: item_def_id },
            cfg: None,
            inline_stmt_id: None,
        }),
    })
}

enum Mode {
    Auto,
    Sized,
}

fn clean_param_env<'tcx>(
    cx: &mut DocContext<'tcx>,
    item_def_id: DefId,
    info: ImplInfo<'tcx>,
) -> clean::Generics {
    let tcx = cx.tcx;
    let generics = tcx.generics_of(item_def_id);

    struct InferReplacer<'tcx> {
        tcx: TyCtxt<'tcx>,
        vid_to_region: FxIndexMap<ty::RegionVid, ty::Region<'tcx>>,
        map: FxIndexMap<Vid, Symbol>,
    }

    impl InferReplacer<'_> {
        // FIXME(fmease): Generate nice names that don't clash with existing params.
        fn name_for(&mut self, vid: Vid) -> Symbol {
            let id = self.map.len();
            *self.map.entry(vid).or_insert_with(|| Symbol::intern(&format!("X{id}")))
        }
    }

    impl<'tcx> TypeFolder<TyCtxt<'tcx>> for InferReplacer<'tcx> {
        fn cx(&self) -> TyCtxt<'tcx> {
            self.tcx
        }

        fn fold_region(&mut self, re: ty::Region<'tcx>) -> ty::Region<'tcx> {
            // FIXME(fmease): Maybe add back != ReErased && != ReLateParam assertion?
            // FIXME(fmease): We can reach RePlaceholder (cc #120606). How to treat?
            let ty::ReVar(vid) = re.kind() else { return re };
            // FIXME(fmease): Generate nice names that don't clash with existing params.
            self.vid_to_region.get(&vid).copied().unwrap_or_else(|| {
                let id = self.map.len();
                ty::Region::new_early_param(
                    self.tcx,
                    ty::EarlyParamRegion {
                        index: u32::MAX,
                        name: *self
                            .map
                            .entry(Vid::Re(vid))
                            .or_insert_with(|| Symbol::intern(&format!("'x{id}"))),
                    },
                )
            })
        }

        fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
            if !ty.has_infer() {
                return ty;
            }
            let &ty::Infer(ty::InferTy::TyVar(vid)) = ty.kind() else {
                return ty.super_fold_with(self);
            };
            Ty::new_param(self.tcx, u32::MAX, self.name_for(Vid::Ty(vid)))
        }

        fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
            if !ct.has_infer() {
                return ct;
            }
            let ty::ConstKind::Infer(ty::InferConst::Var(vid)) = ct.kind() else {
                return ct.super_fold_with(self);
            };
            ty::Const::new_param(
                self.tcx,
                // FIXME: index gets leaked by one of our printers
                ty::ParamConst::new(u32::MAX, self.name_for(Vid::Ct(vid))),
            )
        }
    }

    #[derive(PartialEq, Eq, Hash)]
    enum Vid {
        Ty(ty::TyVid),
        Ct(ty::ConstVid),
        Re(ty::RegionVid),
    }

    let mut replacer =
        InferReplacer { tcx, vid_to_region: info.vid_to_region, map: FxIndexMap::default() };
    let preds = info.param_env.caller_bounds().fold_with(&mut replacer);

    let mut params: ThinVec<_> = generics
        .own_params
        .iter()
        // We're basing the generics of the synthetic auto trait impl off of the generics of the
        // implementing type. Its generic parameters may have defaults, don't copy them over:
        // Generic parameter defaults are meaningless in impls.
        .map(|param| clean_generic_param_def(param, clean::ParamDefaults::No, cx))
        .collect();

    // FIXME(fmease): Move synthetic lifetime params before type/const params.
    params.extend(replacer.map.into_iter().map(|(vid, name)| clean::GenericParamDef {
        name,
        // FIXME(fmease): Make `GPD.def_id` optional & set it to `None` if possible.
        def_id: rustc_hir::def_id::CRATE_DEF_ID.into(),
        kind: match vid {
            Vid::Ty(_) => clean::GenericParamDefKind::Type {
                bounds: ThinVec::new(),
                default: None,
                synthetic: false,
            },
            Vid::Ct(vid) => clean::GenericParamDefKind::Const {
                ty: Box::new(clean_middle_ty(info.const_var_tys[&vid], cx, None, None)),
                default: None,
            },
            Vid::Re(_) => clean::GenericParamDefKind::Lifetime { outlives: ThinVec::new() },
        },
    }));

    // FIXME(#111101): Incorporate the explicit predicates of the item here...
    let item_predicates: FxIndexSet<_> =
        tcx.param_env(item_def_id).caller_bounds().iter().collect();
    let where_predicates = preds
        .iter()
        // FIXME: ...which hopefully allows us to simplify this:
        // FIXME(fmease): Do the filteriing before InferReplacer?
        .filter(|pred| {
            !item_predicates.contains(pred)
                || pred
                    .as_trait_clause()
                    .is_some_and(|pred| tcx.lang_items().sized_trait() == Some(pred.def_id()))
        })
        .flat_map(|pred| clean_predicate(pred, cx))
        .chain(clean_region_outlives_constraints(&info.region_data, generics))
        .collect();

    let mut generics = clean::Generics { params, where_predicates };
    simplify::sized_bounds(cx, &mut generics);
    generics.where_predicates = simplify::where_clauses(cx, generics.where_predicates);
    generics
}

/// Clean region outlives constraints to where-predicates.
///
/// This is essentially a simplified version of `lexical_region_resolve`.
///
/// However, here we determine what *needs to be* true in order for an impl to hold.
/// `lexical_region_resolve`, along with much of the rest of the compiler, is concerned
/// with determining if a given set up constraints / predicates *are* met, given some
/// starting conditions like user-provided code.
///
/// For this reason, it's easier to perform the calculations we need on our own,
/// rather than trying to make existing inference/solver code do what we want.
fn clean_region_outlives_constraints<'tcx>(
    regions: &RegionConstraintData<'tcx>,
    generics: &'tcx ty::Generics,
) -> ThinVec<clean::WherePredicate> {
    // Our goal is to "flatten" the list of constraints by eliminating all intermediate
    // `RegionVids` (region inference variables). At the end, all constraints should be
    // between `Region`s. This gives us the information we need to create the where-predicates.
    // This flattening is done in two parts.

    let mut outlives_predicates = FxIndexMap::<_, Vec<_>>::default();
    let mut map = FxIndexMap::<RegionTarget<'_>, RegionDeps<'_>>::default();

    // (1)  We insert all of the constraints into a map.
    // Each `RegionTarget` (a `RegionVid` or a `Region`) maps to its smaller and larger regions.
    // Note that "larger" regions correspond to sub regions in the surface language.
    // E.g., in `'a: 'b`, `'a` is the larger region.
    for (c, _) in &regions.constraints {
        match c.kind {
            ConstraintKind::VarSubVar => {
                let sub_vid = c.sub.as_var();
                let sup_vid = c.sup.as_var();
                let deps1 = map.entry(RegionTarget::RegionVid(sub_vid)).or_default();
                deps1.larger.insert(RegionTarget::RegionVid(sup_vid));

                let deps2 = map.entry(RegionTarget::RegionVid(sup_vid)).or_default();
                deps2.smaller.insert(RegionTarget::RegionVid(sub_vid));
            }
            ConstraintKind::RegSubVar => {
                let sup_vid = c.sup.as_var();
                let deps = map.entry(RegionTarget::RegionVid(sup_vid)).or_default();
                deps.smaller.insert(RegionTarget::Region(c.sub));
            }
            ConstraintKind::VarSubReg => {
                let sub_vid = c.sub.as_var();
                let deps = map.entry(RegionTarget::RegionVid(sub_vid)).or_default();
                deps.larger.insert(RegionTarget::Region(c.sup));
            }
            ConstraintKind::RegSubReg => {
                // The constraint is already in the form that we want, so we're done with it
                // The desired order is [larger, smaller], so flip them.
                if early_bound_region_name(c.sub) != early_bound_region_name(c.sup) {
                    outlives_predicates
                        .entry(early_bound_region_name(c.sup).expect("no region_name found"))
                        .or_default()
                        .push(c.sub);
                }
            }
        }
    }

    // (2)  Here, we "flatten" the map one element at a time. All of the elements' sub and super
    // regions are connected to each other. For example, if we have a graph that looks like this:
    //
    //     (A, B) - C - (D, E)
    //
    // where (A, B) are sub regions, and (D,E) are super regions.
    // Then, after deleting 'C', the graph will look like this:
    //
    //             ... - A - (D, E, ...)
    //             ... - B - (D, E, ...)
    //     (A, B, ...) - D - ...
    //     (A, B, ...) - E - ...
    //
    // where '...' signifies the existing sub and super regions of an entry. When two adjacent
    // `Region`s are encountered, we've computed a final constraint, and add it to our list.
    // Since we make sure to never re-add deleted items, this process will always finish.
    while !map.is_empty() {
        let target = *map.keys().next().unwrap();
        let deps = map.swap_remove(&target).unwrap();

        for smaller in &deps.smaller {
            for larger in &deps.larger {
                match (smaller, larger) {
                    (&RegionTarget::Region(smaller), &RegionTarget::Region(larger)) => {
                        if early_bound_region_name(smaller) != early_bound_region_name(larger) {
                            outlives_predicates
                                .entry(
                                    early_bound_region_name(larger).expect("no region name found"),
                                )
                                .or_default()
                                .push(smaller)
                        }
                    }
                    (&RegionTarget::RegionVid(_), &RegionTarget::Region(_)) => {
                        if let IndexEntry::Occupied(v) = map.entry(*smaller) {
                            let smaller_deps = v.into_mut();
                            smaller_deps.larger.insert(*larger);
                            smaller_deps.larger.swap_remove(&target);
                        }
                    }
                    (&RegionTarget::Region(_), &RegionTarget::RegionVid(_)) => {
                        if let IndexEntry::Occupied(v) = map.entry(*larger) {
                            let deps = v.into_mut();
                            deps.smaller.insert(*smaller);
                            deps.smaller.swap_remove(&target);
                        }
                    }
                    (&RegionTarget::RegionVid(_), &RegionTarget::RegionVid(_)) => {
                        if let IndexEntry::Occupied(v) = map.entry(*smaller) {
                            let smaller_deps = v.into_mut();
                            smaller_deps.larger.insert(*larger);
                            smaller_deps.larger.swap_remove(&target);
                        }
                        if let IndexEntry::Occupied(v) = map.entry(*larger) {
                            let larger_deps = v.into_mut();
                            larger_deps.smaller.insert(*smaller);
                            larger_deps.smaller.swap_remove(&target);
                        }
                    }
                }
            }
        }
    }

    let region_params: FxIndexSet<_> = generics
        .own_params
        .iter()
        .filter_map(|param| match param.kind {
            ty::GenericParamDefKind::Lifetime => Some(param.name),
            _ => None,
        })
        .collect();

    region_params
        .iter()
        .filter_map(|&name| {
            let bounds: FxIndexSet<_> = outlives_predicates
                .get(&name)?
                .iter()
                .map(|&region| {
                    let lifetime = early_bound_region_name(region)
                        .inspect(|name| assert!(region_params.contains(name)))
                        .map(Lifetime)
                        .unwrap_or(Lifetime::statik());
                    clean::GenericBound::Outlives(lifetime)
                })
                .collect();
            if bounds.is_empty() {
                return None;
            }
            Some(clean::WherePredicate::RegionPredicate {
                lifetime: Lifetime(name),
                bounds: bounds.into_iter().collect(),
            })
        })
        .collect()
}

fn early_bound_region_name(region: Region<'_>) -> Option<Symbol> {
    match region.kind() {
        ty::ReEarlyParam(r) => Some(r.name),
        _ => None,
    }
}

struct PredicateCollector<'tcx> {
    clauses: Vec<ty::Clause<'tcx>>,
    const_var_tys: UnordMap<ty::ConstVid, ty::Binder<'tcx, Ty<'tcx>>>,
}

impl<'tcx> PredicateCollector<'tcx> {
    fn add(&mut self, goal: Goal<'tcx, ty::Predicate<'tcx>>) {
        if let Some(clause) = goal.predicate.as_clause()
            && let Some(_) = clause.as_trait_clause()
        {
            self.clauses.push(clause);
        } else {
            panic!("can't handle PredicateKind {:?}", goal.predicate)
        }
    }
}

impl<'tcx> ProofTreeVisitor<'tcx> for PredicateCollector<'tcx> {
    type Result = ControlFlow<()>;

    fn span(&self) -> rustc_span::Span {
        rustc_span::DUMMY_SP
    }

    fn visit_goal(&mut self, goal: &InspectGoal<'_, 'tcx>) -> Self::Result {
        let tcx = goal.infcx().tcx;
        let predicate = goal.goal().predicate;

        match goal.result() {
            Err(_) => {}
            Ok(certainty) => {
                // // FIXME: HACK
                if let ty::solve::Certainty::Yes = certainty
                    && let Some(clause) = predicate.as_clause()
                    && let ty::ClauseKind::RegionOutlives(_) | ty::ClauseKind::TypeOutlives(_) = clause.kind().skip_binder()
                {
                    self.clauses.push(clause);
                } else

                if let ty::solve::Certainty::AMBIGUOUS = certainty
                    && let Some(clause) = predicate.as_clause()
                {
                    let kind = clause.kind();
                    match kind.skip_binder() {
                        ty::ClauseKind::ConstArgHasType(ct, ty)
                            if let ty::ConstKind::Infer(ty::InferConst::Var(vid)) = ct.kind() =>
                        {
                            self.const_var_tys.insert(vid, kind.rebind(ty));
                        }
                        ty::ClauseKind::Trait(pred) if pred.has_infer() => {
                            self.clauses.push(clause);
                        }
                        ty::ClauseKind::Projection(pred) if pred.has_infer() => {
                            self.clauses.push(clause);
                        }
                        // FIXME: HostEffect
                        _ => {}
                    }
                }

                return ControlFlow::Continue(());
            }
        }

        let candidates = goal.candidates();
        let candidate = match candidates.as_slice() {
            [] => {
                // FIXME: legacy SATI also handles outlives.predicates.
                //        Do we really need to handle them, too?
                if let Some(clause) = predicate.as_clause() {
                    match clause.kind().skip_binder() {
                        ty::ClauseKind::Trait(pred) => {
                            if is_interesting(pred.self_ty()) {
                                self.clauses.push(clause);
                                return ControlFlow::Continue(());
                            }
                        }
                        // NOTE(const_trait_impl): We currently don't display bound constness in rustdoc.
                        ty::ClauseKind::HostEffect(_) => {
                            return ControlFlow::Continue(());
                        }
                        _ => {}
                    }
                }

                return ControlFlow::Break(());
            }
            [candidate] => candidate,
            // FIXME: Is this even reachable?
            _ => {
                self.add(goal.goal());
                return ControlFlow::Continue(());
            }
        };

        // FIXME: Name
        fn is_interesting(ty: Ty<'_>) -> bool {
            match ty.kind() {
                ty::Param(_) => true,
                ty::Alias(ty::Projection, alias_ty) => is_interesting(alias_ty.self_ty()),
                _ => false,
            }
        }

        if let Some(clause) = predicate.as_clause()
            && let ty::ClauseKind::Projection(pred) = clause.kind().skip_binder()
            && is_interesting(pred.self_ty())
        {
            self.clauses.push(clause);
            return ControlFlow::Continue(());
        }

        // FIXME: HACK:
        if let ty::PredicateKind::AliasRelate(term0, term1, ty::AliasRelationDirection::Equate) =
            predicate.kind().skip_binder()
            && let Some(ty0) = term0.as_type()
            && let &ty::Alias(ty::Projection, alias_ty) = ty0.kind()
        {
            self.clauses.push(
                ty::ProjectionPredicate { projection_term: alias_ty.into(), term: term1 }
                    .upcast(tcx),
            );
            return ControlFlow::Continue(());
        }

        let nested_goals = candidate.instantiate_nested_goals(self.span());

        // FIXME(fmease): Explainer
        if nested_goals.is_empty() {
            return ControlFlow::Break(());
        }

        // FIXME(fmease): Explainer.
        if nested_goals.iter().any(|nested_goal| {
            nested_goal
                .goal()
                .predicate
                .as_trait_clause()
                .is_some_and(|clause| tcx.is_lang_item(clause.def_id(), hir::LangItem::FnPtrTrait))
        }) {
            self.add(goal.goal());
            return ControlFlow::Continue(());
        }

        nested_goals.into_iter().try_for_each(|goal| goal.visit_with(self))
    }
}

/// This is very similar to `handle_lifetimes`. However, instead of matching `ty::Region`s
/// to each other, we match `ty::RegionVid`s to `ty::Region`s.
fn map_vid_to_region<'cx>(
    regions: &RegionConstraintData<'cx>,
) -> FxIndexMap<ty::RegionVid, ty::Region<'cx>> {
    let mut vid_map = FxIndexMap::<RegionTarget<'cx>, RegionDeps<'cx>>::default();
    let mut finished_map = FxIndexMap::default();

    for (c, _) in &regions.constraints {
        match c.kind {
            ConstraintKind::VarSubVar => {
                let sub_vid = c.sub.as_var();
                let sup_vid = c.sup.as_var();
                {
                    let deps1 = vid_map.entry(RegionTarget::RegionVid(sub_vid)).or_default();
                    deps1.larger.insert(RegionTarget::RegionVid(sup_vid));
                }

                let deps2 = vid_map.entry(RegionTarget::RegionVid(sup_vid)).or_default();
                deps2.smaller.insert(RegionTarget::RegionVid(sub_vid));
            }
            ConstraintKind::RegSubVar => {
                let sup_vid = c.sup.as_var();
                {
                    let deps1 = vid_map.entry(RegionTarget::Region(c.sub)).or_default();
                    deps1.larger.insert(RegionTarget::RegionVid(sup_vid));
                }

                let deps2 = vid_map.entry(RegionTarget::RegionVid(sup_vid)).or_default();
                deps2.smaller.insert(RegionTarget::Region(c.sub));
            }
            ConstraintKind::VarSubReg => {
                let sub_vid = c.sub.as_var();
                finished_map.insert(sub_vid, c.sup);
            }
            ConstraintKind::RegSubReg => {
                {
                    let deps1 = vid_map.entry(RegionTarget::Region(c.sub)).or_default();
                    deps1.larger.insert(RegionTarget::Region(c.sup));
                }

                let deps2 = vid_map.entry(RegionTarget::Region(c.sup)).or_default();
                deps2.smaller.insert(RegionTarget::Region(c.sub));
            }
        }
    }

    while !vid_map.is_empty() {
        let target = *vid_map.keys().next().unwrap();
        let deps = vid_map.swap_remove(&target).unwrap();

        for smaller in deps.smaller.iter() {
            for larger in deps.larger.iter() {
                match (smaller, larger) {
                    (&RegionTarget::Region(_), &RegionTarget::Region(_)) => {
                        if let IndexEntry::Occupied(v) = vid_map.entry(*smaller) {
                            let smaller_deps = v.into_mut();
                            smaller_deps.larger.insert(*larger);
                            smaller_deps.larger.swap_remove(&target);
                        }

                        if let IndexEntry::Occupied(v) = vid_map.entry(*larger) {
                            let larger_deps = v.into_mut();
                            larger_deps.smaller.insert(*smaller);
                            larger_deps.smaller.swap_remove(&target);
                        }
                    }
                    (&RegionTarget::RegionVid(v1), &RegionTarget::Region(r1)) => {
                        finished_map.insert(v1, r1);
                    }
                    (&RegionTarget::Region(_), &RegionTarget::RegionVid(_)) => {
                        // Do nothing; we don't care about regions that are smaller than vids.
                    }
                    (&RegionTarget::RegionVid(_), &RegionTarget::RegionVid(_)) => {
                        if let IndexEntry::Occupied(v) = vid_map.entry(*smaller) {
                            let smaller_deps = v.into_mut();
                            smaller_deps.larger.insert(*larger);
                            smaller_deps.larger.swap_remove(&target);
                        }

                        if let IndexEntry::Occupied(v) = vid_map.entry(*larger) {
                            let larger_deps = v.into_mut();
                            larger_deps.smaller.insert(*smaller);
                            larger_deps.smaller.swap_remove(&target);
                        }
                    }
                }
            }
        }
    }

    finished_map
}

#[derive(Debug)]
struct ImplInfo<'tcx> {
    param_env: ty::ParamEnv<'tcx>,
    const_var_tys: UnordMap<ty::ConstVid, ty::Binder<'tcx, Ty<'tcx>>>,
    region_data: RegionConstraintData<'tcx>,
    vid_to_region: FxIndexMap<ty::RegionVid, ty::Region<'tcx>>,
}

#[derive(Eq, PartialEq, Hash, Copy, Clone, Debug)]
enum RegionTarget<'tcx> {
    Region(Region<'tcx>),
    RegionVid(ty::RegionVid),
}

#[derive(Default, Debug, Clone)]
struct RegionDeps<'tcx> {
    larger: FxIndexSet<RegionTarget<'tcx>>,
    smaller: FxIndexSet<RegionTarget<'tcx>>,
}
