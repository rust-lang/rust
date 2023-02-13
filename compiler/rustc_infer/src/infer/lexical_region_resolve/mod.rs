//! Lexical region resolution.

use crate::infer::region_constraints::Constraint;
use crate::infer::region_constraints::GenericKind;
use crate::infer::region_constraints::RegionConstraintData;
use crate::infer::region_constraints::VarInfos;
use crate::infer::region_constraints::VerifyBound;
use crate::infer::RegionRelations;
use crate::infer::RegionVariableOrigin;
use crate::infer::SubregionOrigin;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::graph::implementation::{
    Direction, Graph, NodeIndex, INCOMING, OUTGOING,
};
use rustc_data_structures::intern::Interned;
use rustc_index::vec::{Idx, IndexVec};
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::PlaceholderRegion;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{ReEarlyBound, ReErased, ReError, ReFree, ReStatic};
use rustc_middle::ty::{ReLateBound, RePlaceholder, ReVar};
use rustc_middle::ty::{Region, RegionVid};
use rustc_span::Span;
use std::fmt;

use super::outlives::test_type_match;

/// This function performs lexical region resolution given a complete
/// set of constraints and variable origins. It performs a fixed-point
/// iteration to find region values which satisfy all constraints,
/// assuming such values can be found. It returns the final values of
/// all the variables as well as a set of errors that must be reported.
#[instrument(level = "debug", skip(region_rels, var_infos, data))]
pub(crate) fn resolve<'tcx>(
    param_env: ty::ParamEnv<'tcx>,
    region_rels: &RegionRelations<'_, 'tcx>,
    var_infos: VarInfos,
    data: RegionConstraintData<'tcx>,
) -> (LexicalRegionResolutions<'tcx>, Vec<RegionResolutionError<'tcx>>) {
    let mut errors = vec![];
    let mut resolver = LexicalResolver { param_env, region_rels, var_infos, data };
    let values = resolver.infer_variable_values(&mut errors);
    (values, errors)
}

/// Contains the result of lexical region resolution. Offers methods
/// to lookup up the final value of a region variable.
#[derive(Clone)]
pub struct LexicalRegionResolutions<'tcx> {
    pub(crate) values: IndexVec<RegionVid, VarValue<'tcx>>,
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum VarValue<'tcx> {
    /// Empty lifetime is for data that is never accessed. We tag the
    /// empty lifetime with a universe -- the idea is that we don't
    /// want `exists<'a> { forall<'b> { 'b: 'a } }` to be satisfiable.
    /// Therefore, the `'empty` in a universe `U` is less than all
    /// regions visible from `U`, but not less than regions not visible
    /// from `U`.
    Empty(ty::UniverseIndex),
    Value(Region<'tcx>),
    ErrorValue,
}

#[derive(Clone, Debug)]
pub enum RegionResolutionError<'tcx> {
    /// `ConcreteFailure(o, a, b)`:
    ///
    /// `o` requires that `a <= b`, but this does not hold
    ConcreteFailure(SubregionOrigin<'tcx>, Region<'tcx>, Region<'tcx>),

    /// `GenericBoundFailure(p, s, a)
    ///
    /// The parameter/associated-type `p` must be known to outlive the lifetime
    /// `a` (but none of the known bounds are sufficient).
    GenericBoundFailure(SubregionOrigin<'tcx>, GenericKind<'tcx>, Region<'tcx>),

    /// `SubSupConflict(v, v_origin, sub_origin, sub_r, sup_origin, sup_r)`:
    ///
    /// Could not infer a value for `v` (which has origin `v_origin`)
    /// because `sub_r <= v` (due to `sub_origin`) but `v <= sup_r` (due to `sup_origin`) and
    /// `sub_r <= sup_r` does not hold.
    SubSupConflict(
        RegionVid,
        RegionVariableOrigin,
        SubregionOrigin<'tcx>,
        Region<'tcx>,
        SubregionOrigin<'tcx>,
        Region<'tcx>,
        Vec<Span>, // All the influences on a given value that didn't meet its constraints.
    ),

    /// Indicates a `'b: 'a` constraint where `'a` is in a universe that
    /// cannot name the placeholder `'b`.
    UpperBoundUniverseConflict(
        RegionVid,
        RegionVariableOrigin,
        ty::UniverseIndex,     // the universe index of the region variable
        SubregionOrigin<'tcx>, // cause of the constraint
        Region<'tcx>,          // the placeholder `'b`
    ),
}

struct RegionAndOrigin<'tcx> {
    region: Region<'tcx>,
    origin: SubregionOrigin<'tcx>,
}

type RegionGraph<'tcx> = Graph<(), Constraint<'tcx>>;

struct LexicalResolver<'cx, 'tcx> {
    param_env: ty::ParamEnv<'tcx>,
    region_rels: &'cx RegionRelations<'cx, 'tcx>,
    var_infos: VarInfos,
    data: RegionConstraintData<'tcx>,
}

impl<'cx, 'tcx> LexicalResolver<'cx, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.region_rels.tcx
    }

    fn infer_variable_values(
        &mut self,
        errors: &mut Vec<RegionResolutionError<'tcx>>,
    ) -> LexicalRegionResolutions<'tcx> {
        let mut var_data = self.construct_var_data();

        if cfg!(debug_assertions) {
            self.dump_constraints();
        }

        let graph = self.construct_graph();
        self.expand_givens(&graph);
        self.expansion(&mut var_data);
        self.collect_errors(&mut var_data, errors);
        self.collect_var_errors(&var_data, &graph, errors);
        var_data
    }

    fn num_vars(&self) -> usize {
        self.var_infos.len()
    }

    /// Initially, the value for all variables is set to `'empty`, the
    /// empty region. The `expansion` phase will grow this larger.
    fn construct_var_data(&self) -> LexicalRegionResolutions<'tcx> {
        LexicalRegionResolutions {
            values: IndexVec::from_fn_n(
                |vid| {
                    let vid_universe = self.var_infos[vid].universe;
                    VarValue::Empty(vid_universe)
                },
                self.num_vars(),
            ),
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn dump_constraints(&self) {
        for (idx, (constraint, _)) in self.data.constraints.iter().enumerate() {
            debug!("Constraint {} => {:?}", idx, constraint);
        }
    }

    fn expand_givens(&mut self, graph: &RegionGraph<'_>) {
        // Givens are a kind of horrible hack to account for
        // constraints like 'c <= '0 that are known to hold due to
        // closure signatures (see the comment above on the `givens`
        // field). They should go away. But until they do, the role
        // of this fn is to account for the transitive nature:
        //
        //     Given 'c <= '0
        //     and   '0 <= '1
        //     then  'c <= '1

        let seeds: Vec<_> = self.data.givens.iter().cloned().collect();
        for (r, vid) in seeds {
            // While all things transitively reachable in the graph
            // from the variable (`'0` in the example above).
            let seed_index = NodeIndex(vid.index() as usize);
            for succ_index in graph.depth_traverse(seed_index, OUTGOING) {
                let succ_index = succ_index.0;

                // The first N nodes correspond to the region
                // variables. Other nodes correspond to constant
                // regions.
                if succ_index < self.num_vars() {
                    let succ_vid = RegionVid::new(succ_index);

                    // Add `'c <= '1`.
                    self.data.givens.insert((r, succ_vid));
                }
            }
        }
    }

    /// Gets the LUb of a given region and the empty region
    fn lub_empty(&self, a_region: Region<'tcx>) -> Result<Region<'tcx>, PlaceholderRegion> {
        match *a_region {
            ReLateBound(..) | ReErased => {
                bug!("cannot relate region: {:?}", a_region);
            }

            ReVar(v_id) => {
                span_bug!(
                    self.var_infos[v_id].origin.span(),
                    "lub invoked with non-concrete regions: {:?}",
                    a_region,
                );
            }

            ReStatic => {
                // nothing lives longer than `'static`
                Ok(self.tcx().lifetimes.re_static)
            }

            ReError(_) => Ok(a_region),

            ReEarlyBound(_) | ReFree(_) => {
                // All empty regions are less than early-bound, free,
                // and scope regions.
                Ok(a_region)
            }

            RePlaceholder(placeholder) => Err(placeholder),
        }
    }

    fn expansion(&self, var_values: &mut LexicalRegionResolutions<'tcx>) {
        // In the first pass, we expand region vids according to constraints we
        // have previously found. In the second pass, we loop through the region
        // vids we expanded and expand *across* region vids (effectively
        // "expanding" new `RegSubVar` constraints).

        // Tracks the `VarSubVar` constraints generated for each region vid. We
        // later use this to expand across vids.
        let mut constraints = IndexVec::from_elem_n(Vec::new(), var_values.values.len());
        // Tracks the changed region vids.
        let mut changes = Vec::new();
        for constraint in self.data.constraints.keys() {
            match *constraint {
                Constraint::RegSubVar(a_region, b_vid) => {
                    let b_data = var_values.value_mut(b_vid);

                    if self.expand_node(a_region, b_vid, b_data) {
                        changes.push(b_vid);
                    }
                }
                Constraint::VarSubVar(a_vid, b_vid) => match *var_values.value(a_vid) {
                    VarValue::ErrorValue => continue,
                    VarValue::Empty(a_universe) => {
                        let b_data = var_values.value_mut(b_vid);

                        let changed = match *b_data {
                            VarValue::Empty(b_universe) => {
                                // Empty regions are ordered according to the universe
                                // they are associated with.
                                let ui = a_universe.min(b_universe);

                                debug!(
                                    "Expanding value of {:?} \
                                    from empty lifetime with universe {:?} \
                                    to empty lifetime with universe {:?}",
                                    b_vid, b_universe, ui
                                );

                                *b_data = VarValue::Empty(ui);
                                true
                            }
                            VarValue::Value(cur_region) => {
                                let lub = match self.lub_empty(cur_region) {
                                    Ok(r) => r,
                                    // If the empty and placeholder regions are in the same universe,
                                    // then the LUB is the Placeholder region (which is the cur_region).
                                    // If they are not in the same universe, the LUB is the Static lifetime.
                                    Err(placeholder) if a_universe == placeholder.universe => {
                                        cur_region
                                    }
                                    Err(_) => self.tcx().lifetimes.re_static,
                                };

                                if lub == cur_region {
                                    false
                                } else {
                                    debug!(
                                        "Expanding value of {:?} from {:?} to {:?}",
                                        b_vid, cur_region, lub
                                    );

                                    *b_data = VarValue::Value(lub);
                                    true
                                }
                            }

                            VarValue::ErrorValue => false,
                        };

                        if changed {
                            changes.push(b_vid);
                        }
                        match b_data {
                            VarValue::Value(Region(Interned(ReStatic, _)))
                            | VarValue::ErrorValue => (),
                            _ => {
                                constraints[a_vid].push((a_vid, b_vid));
                                constraints[b_vid].push((a_vid, b_vid));
                            }
                        }
                    }
                    VarValue::Value(a_region) => {
                        let b_data = var_values.value_mut(b_vid);

                        if self.expand_node(a_region, b_vid, b_data) {
                            changes.push(b_vid);
                        }
                        match b_data {
                            VarValue::Value(Region(Interned(ReStatic, _)))
                            | VarValue::ErrorValue => (),
                            _ => {
                                constraints[a_vid].push((a_vid, b_vid));
                                constraints[b_vid].push((a_vid, b_vid));
                            }
                        }
                    }
                },
                Constraint::RegSubReg(..) | Constraint::VarSubReg(..) => {
                    // These constraints are checked after expansion
                    // is done, in `collect_errors`.
                    continue;
                }
            }
        }

        while let Some(vid) = changes.pop() {
            constraints[vid].retain(|&(a_vid, b_vid)| {
                let VarValue::Value(a_region) = *var_values.value(a_vid) else {
                    return false;
                };
                let b_data = var_values.value_mut(b_vid);
                if self.expand_node(a_region, b_vid, b_data) {
                    changes.push(b_vid);
                }
                !matches!(
                    b_data,
                    VarValue::Value(Region(Interned(ReStatic, _))) | VarValue::ErrorValue
                )
            });
        }
    }

    /// Expands the value of the region represented with `b_vid` with current
    /// value `b_data` to the lub of `b_data` and `a_region`. The corresponds
    /// with the constraint `'?b: 'a` (`'a <: '?b`), where `'a` is some known
    /// region and `'?b` is some region variable.
    fn expand_node(
        &self,
        a_region: Region<'tcx>,
        b_vid: RegionVid,
        b_data: &mut VarValue<'tcx>,
    ) -> bool {
        debug!("expand_node({:?}, {:?} == {:?})", a_region, b_vid, b_data);

        match *a_region {
            // Check if this relationship is implied by a given.
            ty::ReEarlyBound(_) | ty::ReFree(_) => {
                if self.data.givens.contains(&(a_region, b_vid)) {
                    debug!("given");
                    return false;
                }
            }

            _ => {}
        }

        match *b_data {
            VarValue::Empty(empty_ui) => {
                let lub = match self.lub_empty(a_region) {
                    Ok(r) => r,
                    // If this empty region is from a universe that can
                    // name the placeholder, then the placeholder is
                    // larger; otherwise, the only ancestor is `'static`.
                    Err(placeholder) if empty_ui.can_name(placeholder.universe) => {
                        self.tcx().mk_re_placeholder(placeholder)
                    }
                    Err(_) => self.tcx().lifetimes.re_static,
                };

                debug!("Expanding value of {:?} from empty lifetime to {:?}", b_vid, lub);

                *b_data = VarValue::Value(lub);
                true
            }
            VarValue::Value(cur_region) => {
                // This is a specialized version of the `lub_concrete_regions`
                // check below for a common case, here purely as an
                // optimization.
                let b_universe = self.var_infos[b_vid].universe;

                let mut lub = self.lub_concrete_regions(a_region, cur_region);
                if lub == cur_region {
                    return false;
                }

                // Watch out for `'b: !1` relationships, where the
                // universe of `'b` can't name the placeholder `!1`. In
                // that case, we have to grow `'b` to be `'static` for the
                // relationship to hold. This is obviously a kind of sub-optimal
                // choice -- in the future, when we incorporate a knowledge
                // of the parameter environment, we might be able to find a
                // tighter bound than `'static`.
                //
                // (This might e.g. arise from being asked to prove `for<'a> { 'b: 'a }`.)
                if let ty::RePlaceholder(p) = *lub && b_universe.cannot_name(p.universe) {
                    lub = self.tcx().lifetimes.re_static;
                }

                debug!("Expanding value of {:?} from {:?} to {:?}", b_vid, cur_region, lub);

                *b_data = VarValue::Value(lub);
                true
            }

            VarValue::ErrorValue => false,
        }
    }

    /// True if `a <= b`.
    fn sub_region_values(&self, a: VarValue<'tcx>, b: VarValue<'tcx>) -> bool {
        match (a, b) {
            // Error region is `'static`
            (VarValue::ErrorValue, _) | (_, VarValue::ErrorValue) => return true,
            (VarValue::Empty(a_ui), VarValue::Empty(b_ui)) => {
                // Empty regions are ordered according to the universe
                // they are associated with.
                a_ui.min(b_ui) == b_ui
            }
            (VarValue::Value(a), VarValue::Empty(_)) => {
                match *a {
                    ReLateBound(..) | ReErased | ReError(_) => {
                        bug!("cannot relate region: {:?}", a);
                    }

                    ReVar(v_id) => {
                        span_bug!(
                            self.var_infos[v_id].origin.span(),
                            "lub_concrete_regions invoked with non-concrete region: {:?}",
                            a
                        );
                    }

                    ReStatic | ReEarlyBound(_) | ReFree(_) => {
                        // nothing lives longer than `'static`

                        // All empty regions are less than early-bound, free,
                        // and scope regions.

                        false
                    }

                    RePlaceholder(_) => {
                        // The LUB is either `a` or `'static`
                        false
                    }
                }
            }
            (VarValue::Empty(a_ui), VarValue::Value(b)) => {
                match *b {
                    ReLateBound(..) | ReErased | ReError(_) => {
                        bug!("cannot relate region: {:?}", b);
                    }

                    ReVar(v_id) => {
                        span_bug!(
                            self.var_infos[v_id].origin.span(),
                            "lub_concrete_regions invoked with non-concrete regions: {:?}",
                            b
                        );
                    }

                    ReStatic | ReEarlyBound(_) | ReFree(_) => {
                        // nothing lives longer than `'static`
                        // All empty regions are less than early-bound, free,
                        // and scope regions.
                        true
                    }

                    RePlaceholder(placeholder) => {
                        // If this empty region is from a universe that can
                        // name the placeholder, then the placeholder is
                        // larger; otherwise, the only ancestor is `'static`.
                        return a_ui.can_name(placeholder.universe);
                    }
                }
            }
            (VarValue::Value(a), VarValue::Value(b)) => self.sub_concrete_regions(a, b),
        }
    }

    /// True if `a <= b`, but not defined over inference variables.
    #[instrument(level = "trace", skip(self))]
    fn sub_concrete_regions(&self, a: Region<'tcx>, b: Region<'tcx>) -> bool {
        let tcx = self.tcx();
        let sub_free_regions = |r1, r2| self.region_rels.free_regions.sub_free_regions(tcx, r1, r2);

        // Check for the case where we know that `'b: 'static` -- in that case,
        // `a <= b` for all `a`.
        let b_free_or_static = b.is_free_or_static();
        if b_free_or_static && sub_free_regions(tcx.lifetimes.re_static, b) {
            return true;
        }

        // If both `a` and `b` are free, consult the declared
        // relationships. Note that this can be more precise than the
        // `lub` relationship defined below, since sometimes the "lub"
        // is actually the `postdom_upper_bound` (see
        // `TransitiveRelation` for more details).
        let a_free_or_static = a.is_free_or_static();
        if a_free_or_static && b_free_or_static {
            return sub_free_regions(a, b);
        }

        // For other cases, leverage the LUB code to find the LUB and
        // check if it is equal to `b`.
        self.lub_concrete_regions(a, b) == b
    }

    /// Returns the least-upper-bound of `a` and `b`; i.e., the
    /// smallest region `c` such that `a <= c` and `b <= c`.
    ///
    /// Neither `a` nor `b` may be an inference variable (hence the
    /// term "concrete regions").
    #[instrument(level = "trace", skip(self), ret)]
    fn lub_concrete_regions(&self, a: Region<'tcx>, b: Region<'tcx>) -> Region<'tcx> {
        match (*a, *b) {
            (ReLateBound(..), _) | (_, ReLateBound(..)) | (ReErased, _) | (_, ReErased) => {
                bug!("cannot relate region: LUB({:?}, {:?})", a, b);
            }

            (ReVar(v_id), _) | (_, ReVar(v_id)) => {
                span_bug!(
                    self.var_infos[v_id].origin.span(),
                    "lub_concrete_regions invoked with non-concrete \
                     regions: {:?}, {:?}",
                    a,
                    b
                );
            }

            (ReError(_), _) => a,

            (_, ReError(_)) => b,

            (ReStatic, _) | (_, ReStatic) => {
                // nothing lives longer than `'static`
                self.tcx().lifetimes.re_static
            }

            (ReEarlyBound(_) | ReFree(_), ReEarlyBound(_) | ReFree(_)) => {
                self.region_rels.lub_free_regions(a, b)
            }

            // For these types, we cannot define any additional
            // relationship:
            (RePlaceholder(..), _) | (_, RePlaceholder(..)) => {
                if a == b {
                    a
                } else {
                    self.tcx().lifetimes.re_static
                }
            }
        }
    }

    /// After expansion is complete, go and check upper bounds (i.e.,
    /// cases where the region cannot grow larger than a fixed point)
    /// and check that they are satisfied.
    #[instrument(skip(self, var_data, errors))]
    fn collect_errors(
        &self,
        var_data: &mut LexicalRegionResolutions<'tcx>,
        errors: &mut Vec<RegionResolutionError<'tcx>>,
    ) {
        for (constraint, origin) in &self.data.constraints {
            debug!(?constraint, ?origin);
            match *constraint {
                Constraint::RegSubVar(..) | Constraint::VarSubVar(..) => {
                    // Expansion will ensure that these constraints hold. Ignore.
                }

                Constraint::RegSubReg(sub, sup) => {
                    if self.sub_concrete_regions(sub, sup) {
                        continue;
                    }

                    debug!(
                        "region error at {:?}: \
                         cannot verify that {:?} <= {:?}",
                        origin, sub, sup
                    );

                    errors.push(RegionResolutionError::ConcreteFailure(
                        (*origin).clone(),
                        sub,
                        sup,
                    ));
                }

                Constraint::VarSubReg(a_vid, b_region) => {
                    let a_data = var_data.value_mut(a_vid);
                    debug!("contraction: {:?} == {:?}, {:?}", a_vid, a_data, b_region);

                    let VarValue::Value(a_region) = *a_data else {
                        continue;
                    };

                    // Do not report these errors immediately:
                    // instead, set the variable value to error and
                    // collect them later.
                    if !self.sub_concrete_regions(a_region, b_region) {
                        debug!(
                            "region error at {:?}: \
                            cannot verify that {:?}={:?} <= {:?}",
                            origin, a_vid, a_region, b_region
                        );
                        *a_data = VarValue::ErrorValue;
                    }
                }
            }
        }

        for verify in &self.data.verifys {
            debug!("collect_errors: verify={:?}", verify);
            let sub = var_data.normalize(self.tcx(), verify.region);

            let verify_kind_ty = verify.kind.to_ty(self.tcx());
            let verify_kind_ty = var_data.normalize(self.tcx(), verify_kind_ty);
            if self.bound_is_met(&verify.bound, var_data, verify_kind_ty, sub) {
                continue;
            }

            debug!(
                "collect_errors: region error at {:?}: \
                 cannot verify that {:?} <= {:?}",
                verify.origin, verify.region, verify.bound
            );

            errors.push(RegionResolutionError::GenericBoundFailure(
                verify.origin.clone(),
                verify.kind,
                sub,
            ));
        }
    }

    /// Go over the variables that were declared to be error variables
    /// and create a `RegionResolutionError` for each of them.
    fn collect_var_errors(
        &self,
        var_data: &LexicalRegionResolutions<'tcx>,
        graph: &RegionGraph<'tcx>,
        errors: &mut Vec<RegionResolutionError<'tcx>>,
    ) {
        debug!("collect_var_errors, var_data = {:#?}", var_data.values);

        // This is the best way that I have found to suppress
        // duplicate and related errors. Basically we keep a set of
        // flags for every node. Whenever an error occurs, we will
        // walk some portion of the graph looking to find pairs of
        // conflicting regions to report to the user. As we walk, we
        // trip the flags from false to true, and if we find that
        // we've already reported an error involving any particular
        // node we just stop and don't report the current error. The
        // idea is to report errors that derive from independent
        // regions of the graph, but not those that derive from
        // overlapping locations.
        let mut dup_vec = IndexVec::from_elem_n(None, self.num_vars());

        for (node_vid, value) in var_data.values.iter_enumerated() {
            match *value {
                VarValue::Empty(_) | VarValue::Value(_) => { /* Inference successful */ }
                VarValue::ErrorValue => {
                    // Inference impossible: this value contains
                    // inconsistent constraints.
                    //
                    // I think that in this case we should report an
                    // error now -- unlike the case above, we can't
                    // wait to see whether the user needs the result
                    // of this variable. The reason is that the mere
                    // existence of this variable implies that the
                    // region graph is inconsistent, whether or not it
                    // is used.
                    //
                    // For example, we may have created a region
                    // variable that is the GLB of two other regions
                    // which do not have a GLB. Even if that variable
                    // is not used, it implies that those two regions
                    // *should* have a GLB.
                    //
                    // At least I think this is true. It may be that
                    // the mere existence of a conflict in a region
                    // variable that is not used is not a problem, so
                    // if this rule starts to create problems we'll
                    // have to revisit this portion of the code and
                    // think hard about it. =) -- nikomatsakis

                    // Obtain the spans for all the places that can
                    // influence the constraints on this value for
                    // richer diagnostics in `static_impl_trait`.

                    self.collect_error_for_expanding_node(graph, &mut dup_vec, node_vid, errors);
                }
            }
        }
    }

    fn construct_graph(&self) -> RegionGraph<'tcx> {
        let num_vars = self.num_vars();

        let mut graph = Graph::new();

        for _ in 0..num_vars {
            graph.add_node(());
        }

        // Issue #30438: two distinct dummy nodes, one for incoming
        // edges (dummy_source) and another for outgoing edges
        // (dummy_sink). In `dummy -> a -> b -> dummy`, using one
        // dummy node leads one to think (erroneously) there exists a
        // path from `b` to `a`. Two dummy nodes sidesteps the issue.
        let dummy_source = graph.add_node(());
        let dummy_sink = graph.add_node(());

        for constraint in self.data.constraints.keys() {
            match *constraint {
                Constraint::VarSubVar(a_id, b_id) => {
                    graph.add_edge(
                        NodeIndex(a_id.index() as usize),
                        NodeIndex(b_id.index() as usize),
                        *constraint,
                    );
                }
                Constraint::RegSubVar(_, b_id) => {
                    graph.add_edge(dummy_source, NodeIndex(b_id.index() as usize), *constraint);
                }
                Constraint::VarSubReg(a_id, _) => {
                    graph.add_edge(NodeIndex(a_id.index() as usize), dummy_sink, *constraint);
                }
                Constraint::RegSubReg(..) => {
                    // this would be an edge from `dummy_source` to
                    // `dummy_sink`; just ignore it.
                }
            }
        }

        graph
    }

    fn collect_error_for_expanding_node(
        &self,
        graph: &RegionGraph<'tcx>,
        dup_vec: &mut IndexVec<RegionVid, Option<RegionVid>>,
        node_idx: RegionVid,
        errors: &mut Vec<RegionResolutionError<'tcx>>,
    ) {
        // Errors in expanding nodes result from a lower-bound that is
        // not contained by an upper-bound.
        let (mut lower_bounds, lower_vid_bounds, lower_dup) =
            self.collect_bounding_regions(graph, node_idx, INCOMING, Some(dup_vec));
        let (mut upper_bounds, _, upper_dup) =
            self.collect_bounding_regions(graph, node_idx, OUTGOING, Some(dup_vec));

        if lower_dup || upper_dup {
            return;
        }

        // We place free regions first because we are special casing
        // SubSupConflict(ReFree, ReFree) when reporting error, and so
        // the user will more likely get a specific suggestion.
        fn region_order_key(x: &RegionAndOrigin<'_>) -> u8 {
            match *x.region {
                ReEarlyBound(_) => 0,
                ReFree(_) => 1,
                _ => 2,
            }
        }
        lower_bounds.sort_by_key(region_order_key);
        upper_bounds.sort_by_key(region_order_key);

        let node_universe = self.var_infos[node_idx].universe;

        for lower_bound in &lower_bounds {
            let effective_lower_bound = if let ty::RePlaceholder(p) = *lower_bound.region {
                if node_universe.cannot_name(p.universe) {
                    self.tcx().lifetimes.re_static
                } else {
                    lower_bound.region
                }
            } else {
                lower_bound.region
            };

            for upper_bound in &upper_bounds {
                if !self.sub_concrete_regions(effective_lower_bound, upper_bound.region) {
                    let origin = self.var_infos[node_idx].origin;
                    debug!(
                        "region inference error at {:?} for {:?}: SubSupConflict sub: {:?} \
                         sup: {:?}",
                        origin, node_idx, lower_bound.region, upper_bound.region
                    );

                    errors.push(RegionResolutionError::SubSupConflict(
                        node_idx,
                        origin,
                        lower_bound.origin.clone(),
                        lower_bound.region,
                        upper_bound.origin.clone(),
                        upper_bound.region,
                        vec![],
                    ));
                    return;
                }
            }
        }

        // If we have a scenario like `exists<'a> { forall<'b> { 'b:
        // 'a } }`, we wind up without any lower-bound -- all we have
        // are placeholders as upper bounds, but the universe of the
        // variable `'a`, or some variable that `'a` has to outlive, doesn't
        // permit those placeholders.
        //
        // We only iterate to find the min, which means it doesn't cause reproducibility issues
        #[allow(rustc::potential_query_instability)]
        let min_universe = lower_vid_bounds
            .into_iter()
            .map(|vid| self.var_infos[vid].universe)
            .min()
            .expect("lower_vid_bounds should at least include `node_idx`");

        for upper_bound in &upper_bounds {
            if let ty::RePlaceholder(p) = *upper_bound.region {
                if min_universe.cannot_name(p.universe) {
                    let origin = self.var_infos[node_idx].origin;
                    errors.push(RegionResolutionError::UpperBoundUniverseConflict(
                        node_idx,
                        origin,
                        min_universe,
                        upper_bound.origin.clone(),
                        upper_bound.region,
                    ));
                    return;
                }
            }
        }

        // Errors in earlier passes can yield error variables without
        // resolution errors here; delay ICE in favor of those errors.
        self.tcx().sess.delay_span_bug(
            self.var_infos[node_idx].origin.span(),
            &format!(
                "collect_error_for_expanding_node() could not find \
                 error for var {:?} in universe {:?}, lower_bounds={:#?}, \
                 upper_bounds={:#?}",
                node_idx, node_universe, lower_bounds, upper_bounds
            ),
        );
    }

    /// Collects all regions that "bound" the variable `orig_node_idx` in the
    /// given direction.
    ///
    /// If `dup_vec` is `Some` it's used to track duplicates between successive
    /// calls of this function.
    ///
    /// The return tuple fields are:
    /// - a list of all concrete regions bounding the given region.
    /// - the set of all region variables bounding the given region.
    /// - a `bool` that's true if the returned region variables overlap with
    ///   those returned by a previous call for another region.
    fn collect_bounding_regions(
        &self,
        graph: &RegionGraph<'tcx>,
        orig_node_idx: RegionVid,
        dir: Direction,
        mut dup_vec: Option<&mut IndexVec<RegionVid, Option<RegionVid>>>,
    ) -> (Vec<RegionAndOrigin<'tcx>>, FxHashSet<RegionVid>, bool) {
        struct WalkState<'tcx> {
            set: FxHashSet<RegionVid>,
            stack: Vec<RegionVid>,
            result: Vec<RegionAndOrigin<'tcx>>,
            dup_found: bool,
        }
        let mut state = WalkState {
            set: Default::default(),
            stack: vec![orig_node_idx],
            result: Vec::new(),
            dup_found: false,
        };
        state.set.insert(orig_node_idx);

        // to start off the process, walk the source node in the
        // direction specified
        process_edges(&self.data, &mut state, graph, orig_node_idx, dir);

        while let Some(node_idx) = state.stack.pop() {
            // check whether we've visited this node on some previous walk
            if let Some(dup_vec) = &mut dup_vec {
                if dup_vec[node_idx].is_none() {
                    dup_vec[node_idx] = Some(orig_node_idx);
                } else if dup_vec[node_idx] != Some(orig_node_idx) {
                    state.dup_found = true;
                }

                debug!(
                    "collect_concrete_regions(orig_node_idx={:?}, node_idx={:?})",
                    orig_node_idx, node_idx
                );
            }

            process_edges(&self.data, &mut state, graph, node_idx, dir);
        }

        let WalkState { result, dup_found, set, .. } = state;
        return (result, set, dup_found);

        fn process_edges<'tcx>(
            this: &RegionConstraintData<'tcx>,
            state: &mut WalkState<'tcx>,
            graph: &RegionGraph<'tcx>,
            source_vid: RegionVid,
            dir: Direction,
        ) {
            debug!("process_edges(source_vid={:?}, dir={:?})", source_vid, dir);

            let source_node_index = NodeIndex(source_vid.index() as usize);
            for (_, edge) in graph.adjacent_edges(source_node_index, dir) {
                match edge.data {
                    Constraint::VarSubVar(from_vid, to_vid) => {
                        let opp_vid = if from_vid == source_vid { to_vid } else { from_vid };
                        if state.set.insert(opp_vid) {
                            state.stack.push(opp_vid);
                        }
                    }

                    Constraint::RegSubVar(region, _) | Constraint::VarSubReg(_, region) => {
                        state.result.push(RegionAndOrigin {
                            region,
                            origin: this.constraints.get(&edge.data).unwrap().clone(),
                        });
                    }

                    Constraint::RegSubReg(..) => panic!(
                        "cannot reach reg-sub-reg edge in region inference \
                         post-processing"
                    ),
                }
            }
        }
    }

    fn bound_is_met(
        &self,
        bound: &VerifyBound<'tcx>,
        var_values: &LexicalRegionResolutions<'tcx>,
        generic_ty: Ty<'tcx>,
        min: ty::Region<'tcx>,
    ) -> bool {
        match bound {
            VerifyBound::IfEq(verify_if_eq_b) => {
                let verify_if_eq_b = var_values.normalize(self.region_rels.tcx, *verify_if_eq_b);
                match test_type_match::extract_verify_if_eq(
                    self.tcx(),
                    self.param_env,
                    &verify_if_eq_b,
                    generic_ty,
                ) {
                    Some(r) => {
                        self.bound_is_met(&VerifyBound::OutlivedBy(r), var_values, generic_ty, min)
                    }

                    None => false,
                }
            }

            VerifyBound::OutlivedBy(r) => {
                let a = match *min {
                    ty::ReVar(rid) => var_values.values[rid],
                    _ => VarValue::Value(min),
                };
                let b = match **r {
                    ty::ReVar(rid) => var_values.values[rid],
                    _ => VarValue::Value(*r),
                };
                self.sub_region_values(a, b)
            }

            VerifyBound::IsEmpty => match *min {
                ty::ReVar(rid) => match var_values.values[rid] {
                    VarValue::ErrorValue => false,
                    VarValue::Empty(_) => true,
                    VarValue::Value(_) => false,
                },
                _ => false,
            },

            VerifyBound::AnyBound(bs) => {
                bs.iter().any(|b| self.bound_is_met(b, var_values, generic_ty, min))
            }

            VerifyBound::AllBounds(bs) => {
                bs.iter().all(|b| self.bound_is_met(b, var_values, generic_ty, min))
            }
        }
    }
}

impl<'tcx> fmt::Debug for RegionAndOrigin<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RegionAndOrigin({:?},{:?})", self.region, self.origin)
    }
}

impl<'tcx> LexicalRegionResolutions<'tcx> {
    fn normalize<T>(&self, tcx: TyCtxt<'tcx>, value: T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        tcx.fold_regions(value, |r, _db| self.resolve_region(tcx, r))
    }

    fn value(&self, rid: RegionVid) -> &VarValue<'tcx> {
        &self.values[rid]
    }

    fn value_mut(&mut self, rid: RegionVid) -> &mut VarValue<'tcx> {
        &mut self.values[rid]
    }

    pub(crate) fn resolve_region(
        &self,
        tcx: TyCtxt<'tcx>,
        r: ty::Region<'tcx>,
    ) -> ty::Region<'tcx> {
        let result = match *r {
            ty::ReVar(rid) => match self.values[rid] {
                VarValue::Empty(_) => r,
                VarValue::Value(r) => r,
                VarValue::ErrorValue => tcx.mk_re_error_misc(),
            },
            _ => r,
        };
        debug!("resolve_region({:?}) = {:?}", r, result);
        result
    }
}
