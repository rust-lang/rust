// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The code to do lexical region resolution.

use infer::SubregionOrigin;
use infer::RegionVariableOrigin;
use infer::region_inference::Constraint;
use infer::region_inference::GenericKind;
use infer::region_inference::RegionVarBindings;
use infer::region_inference::VarValue;
use infer::region_inference::VerifyBound;
use middle::free_region::RegionRelations;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::graph::{self, Direction, NodeIndex, OUTGOING};
use std::fmt;
use std::u32;
use ty::{self, TyCtxt};
use ty::{Region, RegionVid};
use ty::{ReEarlyBound, ReEmpty, ReErased, ReFree, ReStatic};
use ty::{ReLateBound, ReScope, ReSkolemized, ReVar};

mod graphviz;

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

    /// `SubSupConflict(v, sub_origin, sub_r, sup_origin, sup_r)`:
    ///
    /// Could not infer a value for `v` because `sub_r <= v` (due to
    /// `sub_origin`) but `v <= sup_r` (due to `sup_origin`) and
    /// `sub_r <= sup_r` does not hold.
    SubSupConflict(
        RegionVariableOrigin,
        SubregionOrigin<'tcx>,
        Region<'tcx>,
        SubregionOrigin<'tcx>,
        Region<'tcx>,
    ),
}

struct RegionAndOrigin<'tcx> {
    region: Region<'tcx>,
    origin: SubregionOrigin<'tcx>,
}

type RegionGraph<'tcx> = graph::Graph<(), Constraint<'tcx>>;

impl<'a, 'gcx, 'tcx> RegionVarBindings<'a, 'gcx, 'tcx> {
    /// This function performs the actual region resolution.  It must be
    /// called after all constraints have been added.  It performs a
    /// fixed-point iteration to find region values which satisfy all
    /// constraints, assuming such values can be found; if they cannot,
    /// errors are reported.
    pub fn resolve_regions(
        &self,
        region_rels: &RegionRelations<'a, 'gcx, 'tcx>,
    ) -> Vec<RegionResolutionError<'tcx>> {
        debug!("RegionVarBindings: resolve_regions()");
        let mut errors = vec![];
        let v = self.infer_variable_values(region_rels, &mut errors);
        *self.values.borrow_mut() = Some(v);
        errors
    }

    pub fn resolve_var(&self, rid: RegionVid) -> ty::Region<'tcx> {
        match *self.values.borrow() {
            None => span_bug!(
                (*self.var_origins.borrow())[rid.index as usize].span(),
                "attempt to resolve region variable before values have \
                 been computed!"
            ),
            Some(ref values) => {
                let r = lookup(self.tcx, values, rid);
                debug!("resolve_var({:?}) = {:?}", rid, r);
                r
            }
        }
    }

    fn lub_concrete_regions(
        &self,
        region_rels: &RegionRelations<'a, 'gcx, 'tcx>,
        a: Region<'tcx>,
        b: Region<'tcx>,
    ) -> Region<'tcx> {
        match (a, b) {
            (&ReLateBound(..), _) | (_, &ReLateBound(..)) | (&ReErased, _) | (_, &ReErased) => {
                bug!("cannot relate region: LUB({:?}, {:?})", a, b);
            }

            (r @ &ReStatic, _) | (_, r @ &ReStatic) => {
                r // nothing lives longer than static
            }

            (&ReEmpty, r) | (r, &ReEmpty) => {
                r // everything lives longer than empty
            }

            (&ReVar(v_id), _) | (_, &ReVar(v_id)) => {
                span_bug!(
                    (*self.var_origins.borrow())[v_id.index as usize].span(),
                    "lub_concrete_regions invoked with non-concrete \
                     regions: {:?}, {:?}",
                    a,
                    b
                );
            }

            (&ReEarlyBound(_), &ReScope(s_id)) |
            (&ReScope(s_id), &ReEarlyBound(_)) |
            (&ReFree(_), &ReScope(s_id)) |
            (&ReScope(s_id), &ReFree(_)) => {
                // A "free" region can be interpreted as "some region
                // at least as big as fr.scope".  So, we can
                // reasonably compare free regions and scopes:
                let fr_scope = match (a, b) {
                    (&ReEarlyBound(ref br), _) | (_, &ReEarlyBound(ref br)) => {
                        region_rels.region_scope_tree.early_free_scope(self.tcx, br)
                    }
                    (&ReFree(ref fr), _) | (_, &ReFree(ref fr)) => {
                        region_rels.region_scope_tree.free_scope(self.tcx, fr)
                    }
                    _ => bug!(),
                };
                let r_id = region_rels
                    .region_scope_tree
                    .nearest_common_ancestor(fr_scope, s_id);
                if r_id == fr_scope {
                    // if the free region's scope `fr.scope` is bigger than
                    // the scope region `s_id`, then the LUB is the free
                    // region itself:
                    match (a, b) {
                        (_, &ReScope(_)) => return a,
                        (&ReScope(_), _) => return b,
                        _ => bug!(),
                    }
                }

                // otherwise, we don't know what the free region is,
                // so we must conservatively say the LUB is static:
                self.tcx.types.re_static
            }

            (&ReScope(a_id), &ReScope(b_id)) => {
                // The region corresponding to an outer block is a
                // subtype of the region corresponding to an inner
                // block.
                let lub = region_rels
                    .region_scope_tree
                    .nearest_common_ancestor(a_id, b_id);
                self.tcx.mk_region(ReScope(lub))
            }

            (&ReEarlyBound(_), &ReEarlyBound(_)) |
            (&ReFree(_), &ReEarlyBound(_)) |
            (&ReEarlyBound(_), &ReFree(_)) |
            (&ReFree(_), &ReFree(_)) => region_rels.lub_free_regions(a, b),

            // For these types, we cannot define any additional
            // relationship:
            (&ReSkolemized(..), _) | (_, &ReSkolemized(..)) => if a == b {
                a
            } else {
                self.tcx.types.re_static
            },
        }
    }

    fn infer_variable_values(
        &self,
        region_rels: &RegionRelations<'a, 'gcx, 'tcx>,
        errors: &mut Vec<RegionResolutionError<'tcx>>,
    ) -> Vec<VarValue<'tcx>> {
        let mut var_data = self.construct_var_data();

        // Dorky hack to cause `dump_constraints` to only get called
        // if debug mode is enabled:
        debug!(
            "----() End constraint listing (context={:?}) {:?}---",
            region_rels.context,
            self.dump_constraints(region_rels)
        );
        graphviz::maybe_print_constraints_for(self, region_rels);

        let graph = self.construct_graph();
        self.expand_givens(&graph);
        self.expansion(region_rels, &mut var_data);
        self.collect_errors(region_rels, &mut var_data, errors);
        self.collect_var_errors(region_rels, &var_data, &graph, errors);
        var_data
    }

    fn construct_var_data(&self) -> Vec<VarValue<'tcx>> {
        (0..self.num_vars() as usize)
            .map(|_| VarValue::Value(self.tcx.types.re_empty))
            .collect()
    }

    fn dump_constraints(&self, free_regions: &RegionRelations<'a, 'gcx, 'tcx>) {
        debug!(
            "----() Start constraint listing (context={:?}) ()----",
            free_regions.context
        );
        for (idx, (constraint, _)) in self.constraints.borrow().iter().enumerate() {
            debug!("Constraint {} => {:?}", idx, constraint);
        }
    }

    fn expand_givens(&self, graph: &RegionGraph) {
        // Givens are a kind of horrible hack to account for
        // constraints like 'c <= '0 that are known to hold due to
        // closure signatures (see the comment above on the `givens`
        // field). They should go away. But until they do, the role
        // of this fn is to account for the transitive nature:
        //
        //     Given 'c <= '0
        //     and   '0 <= '1
        //     then  'c <= '1

        let mut givens = self.givens.borrow_mut();
        let seeds: Vec<_> = givens.iter().cloned().collect();
        for (r, vid) in seeds {
            let seed_index = NodeIndex(vid.index as usize);
            for succ_index in graph.depth_traverse(seed_index, OUTGOING) {
                let succ_index = succ_index.0 as u32;
                if succ_index < self.num_vars() {
                    let succ_vid = RegionVid { index: succ_index };
                    givens.insert((r, succ_vid));
                }
            }
        }
    }

    fn expansion(
        &self,
        region_rels: &RegionRelations<'a, 'gcx, 'tcx>,
        var_values: &mut [VarValue<'tcx>],
    ) {
        self.iterate_until_fixed_point("Expansion", |constraint, origin| {
            debug!("expansion: constraint={:?} origin={:?}", constraint, origin);
            match *constraint {
                Constraint::RegSubVar(a_region, b_vid) => {
                    let b_data = &mut var_values[b_vid.index as usize];
                    self.expand_node(region_rels, a_region, b_vid, b_data)
                }
                Constraint::VarSubVar(a_vid, b_vid) => match var_values[a_vid.index as usize] {
                    VarValue::ErrorValue => false,
                    VarValue::Value(a_region) => {
                        let b_node = &mut var_values[b_vid.index as usize];
                        self.expand_node(region_rels, a_region, b_vid, b_node)
                    }
                },
                Constraint::RegSubReg(..) | Constraint::VarSubReg(..) => {
                    // These constraints are checked after expansion
                    // is done, in `collect_errors`.
                    false
                }
            }
        })
    }

    fn expand_node(
        &self,
        region_rels: &RegionRelations<'a, 'gcx, 'tcx>,
        a_region: Region<'tcx>,
        b_vid: RegionVid,
        b_data: &mut VarValue<'tcx>,
    ) -> bool {
        debug!("expand_node({:?}, {:?} == {:?})", a_region, b_vid, b_data);

        // Check if this relationship is implied by a given.
        match *a_region {
            ty::ReEarlyBound(_) | ty::ReFree(_) => {
                if self.givens.borrow().contains(&(a_region, b_vid)) {
                    debug!("given");
                    return false;
                }
            }
            _ => {}
        }

        match *b_data {
            VarValue::Value(cur_region) => {
                let lub = self.lub_concrete_regions(region_rels, a_region, cur_region);
                if lub == cur_region {
                    return false;
                }

                debug!(
                    "Expanding value of {:?} from {:?} to {:?}",
                    b_vid,
                    cur_region,
                    lub
                );

                *b_data = VarValue::Value(lub);
                return true;
            }

            VarValue::ErrorValue => {
                return false;
            }
        }
    }

    /// After expansion is complete, go and check upper bounds (i.e.,
    /// cases where the region cannot grow larger than a fixed point)
    /// and check that they are satisfied.
    fn collect_errors(
        &self,
        region_rels: &RegionRelations<'a, 'gcx, 'tcx>,
        var_data: &mut Vec<VarValue<'tcx>>,
        errors: &mut Vec<RegionResolutionError<'tcx>>,
    ) {
        let constraints = self.constraints.borrow();
        for (constraint, origin) in constraints.iter() {
            debug!(
                "collect_errors: constraint={:?} origin={:?}",
                constraint,
                origin
            );
            match *constraint {
                Constraint::RegSubVar(..) | Constraint::VarSubVar(..) => {
                    // Expansion will ensure that these constraints hold. Ignore.
                }

                Constraint::RegSubReg(sub, sup) => {
                    if region_rels.is_subregion_of(sub, sup) {
                        continue;
                    }

                    debug!(
                        "collect_errors: region error at {:?}: \
                         cannot verify that {:?} <= {:?}",
                        origin,
                        sub,
                        sup
                    );

                    errors.push(RegionResolutionError::ConcreteFailure(
                        (*origin).clone(),
                        sub,
                        sup,
                    ));
                }

                Constraint::VarSubReg(a_vid, b_region) => {
                    let a_data = &mut var_data[a_vid.index as usize];
                    debug!("contraction: {:?} == {:?}, {:?}", a_vid, a_data, b_region);

                    let a_region = match *a_data {
                        VarValue::ErrorValue => continue,
                        VarValue::Value(a_region) => a_region,
                    };

                    // Do not report these errors immediately:
                    // instead, set the variable value to error and
                    // collect them later.
                    if !region_rels.is_subregion_of(a_region, b_region) {
                        debug!(
                            "collect_errors: region error at {:?}: \
                             cannot verify that {:?}={:?} <= {:?}",
                            origin,
                            a_vid,
                            a_region,
                            b_region
                        );
                        *a_data = VarValue::ErrorValue;
                    }
                }
            }
        }

        for verify in self.verifys.borrow().iter() {
            debug!("collect_errors: verify={:?}", verify);
            let sub = normalize(self.tcx, var_data, verify.region);

            // This was an inference variable which didn't get
            // constrained, therefore it can be assume to hold.
            if let ty::ReEmpty = *sub {
                continue;
            }

            if verify.bound.is_met(region_rels, var_data, sub) {
                continue;
            }

            debug!(
                "collect_errors: region error at {:?}: \
                 cannot verify that {:?} <= {:?}",
                verify.origin,
                verify.region,
                verify.bound
            );

            errors.push(RegionResolutionError::GenericBoundFailure(
                verify.origin.clone(),
                verify.kind.clone(),
                sub,
            ));
        }
    }

    /// Go over the variables that were declared to be error variables
    /// and create a `RegionResolutionError` for each of them.
    fn collect_var_errors(
        &self,
        region_rels: &RegionRelations<'a, 'gcx, 'tcx>,
        var_data: &[VarValue<'tcx>],
        graph: &RegionGraph<'tcx>,
        errors: &mut Vec<RegionResolutionError<'tcx>>,
    ) {
        debug!("collect_var_errors");

        // This is the best way that I have found to suppress
        // duplicate and related errors. Basically we keep a set of
        // flags for every node. Whenever an error occurs, we will
        // walk some portion of the graph looking to find pairs of
        // conflicting regions to report to the user. As we walk, we
        // trip the flags from false to true, and if we find that
        // we've already reported an error involving any particular
        // node we just stop and don't report the current error.  The
        // idea is to report errors that derive from independent
        // regions of the graph, but not those that derive from
        // overlapping locations.
        let mut dup_vec = vec![u32::MAX; self.num_vars() as usize];

        for idx in 0..self.num_vars() as usize {
            match var_data[idx] {
                VarValue::Value(_) => { /* Inference successful */ }
                VarValue::ErrorValue => {
                    /* Inference impossible, this value contains
                       inconsistent constraints.

                       I think that in this case we should report an
                       error now---unlike the case above, we can't
                       wait to see whether the user needs the result
                       of this variable.  The reason is that the mere
                       existence of this variable implies that the
                       region graph is inconsistent, whether or not it
                       is used.

                       For example, we may have created a region
                       variable that is the GLB of two other regions
                       which do not have a GLB.  Even if that variable
                       is not used, it implies that those two regions
                       *should* have a GLB.

                       At least I think this is true. It may be that
                       the mere existence of a conflict in a region variable
                       that is not used is not a problem, so if this rule
                       starts to create problems we'll have to revisit
                       this portion of the code and think hard about it. =) */

                    let node_vid = RegionVid { index: idx as u32 };
                    self.collect_error_for_expanding_node(
                        region_rels,
                        graph,
                        &mut dup_vec,
                        node_vid,
                        errors,
                    );
                }
            }
        }
    }

    fn construct_graph(&self) -> RegionGraph<'tcx> {
        let num_vars = self.num_vars();

        let constraints = self.constraints.borrow();

        let mut graph = graph::Graph::new();

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

        for (constraint, _) in constraints.iter() {
            match *constraint {
                Constraint::VarSubVar(a_id, b_id) => {
                    graph.add_edge(
                        NodeIndex(a_id.index as usize),
                        NodeIndex(b_id.index as usize),
                        *constraint,
                    );
                }
                Constraint::RegSubVar(_, b_id) => {
                    graph.add_edge(dummy_source, NodeIndex(b_id.index as usize), *constraint);
                }
                Constraint::VarSubReg(a_id, _) => {
                    graph.add_edge(NodeIndex(a_id.index as usize), dummy_sink, *constraint);
                }
                Constraint::RegSubReg(..) => {
                    // this would be an edge from `dummy_source` to
                    // `dummy_sink`; just ignore it.
                }
            }
        }

        return graph;
    }

    fn collect_error_for_expanding_node(
        &self,
        region_rels: &RegionRelations<'a, 'gcx, 'tcx>,
        graph: &RegionGraph<'tcx>,
        dup_vec: &mut [u32],
        node_idx: RegionVid,
        errors: &mut Vec<RegionResolutionError<'tcx>>,
    ) {
        // Errors in expanding nodes result from a lower-bound that is
        // not contained by an upper-bound.
        let (mut lower_bounds, lower_dup) =
            self.collect_concrete_regions(graph, node_idx, graph::INCOMING, dup_vec);
        let (mut upper_bounds, upper_dup) =
            self.collect_concrete_regions(graph, node_idx, graph::OUTGOING, dup_vec);

        if lower_dup || upper_dup {
            return;
        }

        // We place free regions first because we are special casing
        // SubSupConflict(ReFree, ReFree) when reporting error, and so
        // the user will more likely get a specific suggestion.
        fn region_order_key(x: &RegionAndOrigin) -> u8 {
            match *x.region {
                ReEarlyBound(_) => 0,
                ReFree(_) => 1,
                _ => 2,
            }
        }
        lower_bounds.sort_by_key(region_order_key);
        upper_bounds.sort_by_key(region_order_key);

        for lower_bound in &lower_bounds {
            for upper_bound in &upper_bounds {
                if !region_rels.is_subregion_of(lower_bound.region, upper_bound.region) {
                    let origin = (*self.var_origins.borrow())[node_idx.index as usize].clone();
                    debug!(
                        "region inference error at {:?} for {:?}: SubSupConflict sub: {:?} \
                         sup: {:?}",
                        origin,
                        node_idx,
                        lower_bound.region,
                        upper_bound.region
                    );
                    errors.push(RegionResolutionError::SubSupConflict(
                        origin,
                        lower_bound.origin.clone(),
                        lower_bound.region,
                        upper_bound.origin.clone(),
                        upper_bound.region,
                    ));
                    return;
                }
            }
        }

        span_bug!(
            (*self.var_origins.borrow())[node_idx.index as usize].span(),
            "collect_error_for_expanding_node() could not find \
             error for var {:?}, lower_bounds={:?}, \
             upper_bounds={:?}",
            node_idx,
            lower_bounds,
            upper_bounds
        );
    }

    fn collect_concrete_regions(
        &self,
        graph: &RegionGraph<'tcx>,
        orig_node_idx: RegionVid,
        dir: Direction,
        dup_vec: &mut [u32],
    ) -> (Vec<RegionAndOrigin<'tcx>>, bool) {
        struct WalkState<'tcx> {
            set: FxHashSet<RegionVid>,
            stack: Vec<RegionVid>,
            result: Vec<RegionAndOrigin<'tcx>>,
            dup_found: bool,
        }
        let mut state = WalkState {
            set: FxHashSet(),
            stack: vec![orig_node_idx],
            result: Vec::new(),
            dup_found: false,
        };
        state.set.insert(orig_node_idx);

        // to start off the process, walk the source node in the
        // direction specified
        process_edges(self, &mut state, graph, orig_node_idx, dir);

        while !state.stack.is_empty() {
            let node_idx = state.stack.pop().unwrap();

            // check whether we've visited this node on some previous walk
            if dup_vec[node_idx.index as usize] == u32::MAX {
                dup_vec[node_idx.index as usize] = orig_node_idx.index;
            } else if dup_vec[node_idx.index as usize] != orig_node_idx.index {
                state.dup_found = true;
            }

            debug!(
                "collect_concrete_regions(orig_node_idx={:?}, node_idx={:?})",
                orig_node_idx,
                node_idx
            );

            process_edges(self, &mut state, graph, node_idx, dir);
        }

        let WalkState {
            result, dup_found, ..
        } = state;
        return (result, dup_found);

        fn process_edges<'a, 'gcx, 'tcx>(
            this: &RegionVarBindings<'a, 'gcx, 'tcx>,
            state: &mut WalkState<'tcx>,
            graph: &RegionGraph<'tcx>,
            source_vid: RegionVid,
            dir: Direction,
        ) {
            debug!("process_edges(source_vid={:?}, dir={:?})", source_vid, dir);

            let source_node_index = NodeIndex(source_vid.index as usize);
            for (_, edge) in graph.adjacent_edges(source_node_index, dir) {
                match edge.data {
                    Constraint::VarSubVar(from_vid, to_vid) => {
                        let opp_vid = if from_vid == source_vid {
                            to_vid
                        } else {
                            from_vid
                        };
                        if state.set.insert(opp_vid) {
                            state.stack.push(opp_vid);
                        }
                    }

                    Constraint::RegSubVar(region, _) | Constraint::VarSubReg(_, region) => {
                        state.result.push(RegionAndOrigin {
                            region,
                            origin: this.constraints.borrow().get(&edge.data).unwrap().clone(),
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

    fn iterate_until_fixed_point<F>(&self, tag: &str, mut body: F)
    where
        F: FnMut(&Constraint<'tcx>, &SubregionOrigin<'tcx>) -> bool,
    {
        let mut iteration = 0;
        let mut changed = true;
        while changed {
            changed = false;
            iteration += 1;
            debug!("---- {} Iteration {}{}", "#", tag, iteration);
            for (constraint, origin) in self.constraints.borrow().iter() {
                let edge_changed = body(constraint, origin);
                if edge_changed {
                    debug!("Updated due to constraint {:?}", constraint);
                    changed = true;
                }
            }
        }
        debug!("---- {} Complete after {} iteration(s)", tag, iteration);
    }
}

fn normalize<'a, 'gcx, 'tcx>(
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    values: &Vec<VarValue<'tcx>>,
    r: ty::Region<'tcx>,
) -> ty::Region<'tcx> {
    match *r {
        ty::ReVar(rid) => lookup(tcx, values, rid),
        _ => r,
    }
}

fn lookup<'a, 'gcx, 'tcx>(
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    values: &Vec<VarValue<'tcx>>,
    rid: ty::RegionVid,
) -> ty::Region<'tcx> {
    match values[rid.index as usize] {
        VarValue::Value(r) => r,
        VarValue::ErrorValue => tcx.types.re_static, // Previously reported error.
    }
}

impl<'tcx> fmt::Debug for RegionAndOrigin<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RegionAndOrigin({:?},{:?})", self.region, self.origin)
    }
}


impl<'a, 'gcx, 'tcx> VerifyBound<'tcx> {
    fn is_met(
        &self,
        region_rels: &RegionRelations<'a, 'gcx, 'tcx>,
        var_values: &Vec<VarValue<'tcx>>,
        min: ty::Region<'tcx>,
    ) -> bool {
        let tcx = region_rels.tcx;
        match self {
            &VerifyBound::AnyRegion(ref rs) => rs.iter()
                .map(|&r| normalize(tcx, var_values, r))
                .any(|r| region_rels.is_subregion_of(min, r)),

            &VerifyBound::AllRegions(ref rs) => rs.iter()
                .map(|&r| normalize(tcx, var_values, r))
                .all(|r| region_rels.is_subregion_of(min, r)),

            &VerifyBound::AnyBound(ref bs) => {
                bs.iter().any(|b| b.is_met(region_rels, var_values, min))
            }

            &VerifyBound::AllBounds(ref bs) => {
                bs.iter().all(|b| b.is_met(region_rels, var_values, min))
            }
        }
    }
}
