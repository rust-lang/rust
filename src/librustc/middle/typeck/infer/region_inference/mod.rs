// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! See doc.rs */


use middle::ty;
use middle::ty::{FreeRegion, Region, RegionVid};
use middle::ty::{ReEmpty, ReStatic, ReInfer, ReFree, ReEarlyBound,
                 ReLateBound};
use middle::ty::{ReScope, ReVar, ReSkolemized, BrFresh};
use middle::typeck::infer::cres;
use middle::typeck::infer::{RegionVariableOrigin, SubregionOrigin};
use middle::typeck::infer;
use middle::graph;
use middle::graph::{Direction, NodeIndex};
use util::common::indenter;
use util::ppaux::{Repr};

use std::hashmap::{HashMap, HashSet};
use std::uint;
use std::vec;
use syntax::ast;
use syntax::opt_vec;
use syntax::opt_vec::OptVec;

mod doc;

#[deriving(Eq, IterBytes)]
enum Constraint {
    ConstrainVarSubVar(RegionVid, RegionVid),
    ConstrainRegSubVar(Region, RegionVid),
    ConstrainVarSubReg(RegionVid, Region),
    ConstrainRegSubReg(Region, Region),
}

#[deriving(Eq, IterBytes)]
struct TwoRegions {
    a: Region,
    b: Region,
}

enum UndoLogEntry {
    Snapshot,
    AddVar(RegionVid),
    AddConstraint(Constraint),
    AddCombination(CombineMapType, TwoRegions)
}

enum CombineMapType {
    Lub, Glb
}

pub enum RegionResolutionError {
    /// `ConcreteFailure(o, a, b)`:
    ///
    /// `o` requires that `a <= b`, but this does not hold
    ConcreteFailure(SubregionOrigin, Region, Region),

    /// `SubSupConflict(v, sub_origin, sub_r, sup_origin, sup_r)`:
    ///
    /// Could not infer a value for `v` because `sub_r <= v` (due to
    /// `sub_origin`) but `v <= sup_r` (due to `sup_origin`) and
    /// `sub_r <= sup_r` does not hold.
    SubSupConflict(RegionVariableOrigin,
                   SubregionOrigin, Region,
                   SubregionOrigin, Region),

    /// `SupSupConflict(v, origin1, r1, origin2, r2)`:
    ///
    /// Could not infer a value for `v` because `v <= r1` (due to
    /// `origin1`) and `v <= r2` (due to `origin2`) and
    /// `r1` and `r2` have no intersection.
    SupSupConflict(RegionVariableOrigin,
                   SubregionOrigin, Region,
                   SubregionOrigin, Region),
}

type CombineMap = HashMap<TwoRegions, RegionVid>;

pub struct RegionVarBindings {
    tcx: ty::ctxt,
    var_origins: ~[RegionVariableOrigin],
    constraints: HashMap<Constraint, SubregionOrigin>,
    lubs: CombineMap,
    glbs: CombineMap,
    skolemization_count: uint,
    bound_count: uint,

    // The undo log records actions that might later be undone.
    //
    // Note: when the undo_log is empty, we are not actively
    // snapshotting.  When the `start_snapshot()` method is called, we
    // push a Snapshot entry onto the list to indicate that we are now
    // actively snapshotting.  The reason for this is that otherwise
    // we end up adding entries for things like the lower bound on
    // a variable and so forth, which can never be rolled back.
    undo_log: ~[UndoLogEntry],

    // This contains the results of inference.  It begins as an empty
    // option and only acquires a value after inference is complete.
    values: Option<~[VarValue]>,
}

pub fn RegionVarBindings(tcx: ty::ctxt) -> RegionVarBindings {
    RegionVarBindings {
        tcx: tcx,
        var_origins: ~[],
        values: None,
        constraints: HashMap::new(),
        lubs: HashMap::new(),
        glbs: HashMap::new(),
        skolemization_count: 0,
        bound_count: 0,
        undo_log: ~[]
    }
}

impl RegionVarBindings {
    pub fn in_snapshot(&self) -> bool {
        self.undo_log.len() > 0
    }

    pub fn start_snapshot(&mut self) -> uint {
        debug!("RegionVarBindings: snapshot()={}", self.undo_log.len());
        if self.in_snapshot() {
            self.undo_log.len()
        } else {
            self.undo_log.push(Snapshot);
            0
        }
    }

    pub fn commit(&mut self) {
        debug!("RegionVarBindings: commit()");
        while self.undo_log.len() > 0 {
            self.undo_log.pop();
        }
    }

    pub fn rollback_to(&mut self, snapshot: uint) {
        debug!("RegionVarBindings: rollback_to({})", snapshot);
        while self.undo_log.len() > snapshot {
            let undo_item = self.undo_log.pop();
            debug!("undo_item={:?}", undo_item);
            match undo_item {
              Snapshot => {}
              AddVar(vid) => {
                assert_eq!(self.var_origins.len(), vid.to_uint() + 1);
                self.var_origins.pop();
              }
              AddConstraint(ref constraint) => {
                self.constraints.remove(constraint);
              }
              AddCombination(Glb, ref regions) => {
                self.glbs.remove(regions);
              }
              AddCombination(Lub, ref regions) => {
                self.lubs.remove(regions);
              }
            }
        }
    }

    pub fn num_vars(&self) -> uint {
        self.var_origins.len()
    }

    pub fn new_region_var(&mut self, origin: RegionVariableOrigin) -> RegionVid {
        let id = self.num_vars();
        self.var_origins.push(origin);
        let vid = RegionVid { id: id };
        if self.in_snapshot() {
            self.undo_log.push(AddVar(vid));
        }
        debug!("created new region variable {:?} with origin {:?}",
               vid, origin.repr(self.tcx));
        return vid;
    }

    pub fn new_skolemized(&mut self, br: ty::BoundRegion) -> Region {
        let sc = self.skolemization_count;
        self.skolemization_count += 1;
        ReInfer(ReSkolemized(sc, br))
    }

    pub fn new_bound(&mut self, binder_id: ast::NodeId) -> Region {
        // Creates a fresh bound variable for use in GLB computations.
        // See discussion of GLB computation in the large comment at
        // the top of this file for more details.
        //
        // This computation is potentially wrong in the face of
        // rollover.  It's conceivable, if unlikely, that one might
        // wind up with accidental capture for nested functions in
        // that case, if the outer function had bound regions created
        // a very long time before and the inner function somehow
        // wound up rolling over such that supposedly fresh
        // identifiers were in fact shadowed. For now, we just assert
        // that there is no rollover -- eventually we should try to be
        // robust against this possibility, either by checking the set
        // of bound identifiers that appear in a given expression and
        // ensure that we generate one that is distinct, or by
        // changing the representation of bound regions in a fn
        // declaration

        let sc = self.bound_count;
        self.bound_count += 1;

        if sc >= self.bound_count {
            self.tcx.sess.bug("Rollover in RegionInference new_bound()");
        }

        ReLateBound(binder_id, BrFresh(sc))
    }

    pub fn add_constraint(&mut self,
                          constraint: Constraint,
                          origin: SubregionOrigin) {
        // cannot add constraints once regions are resolved
        assert!(self.values.is_none());

        debug!("RegionVarBindings: add_constraint({:?})", constraint);

        if self.constraints.insert(constraint, origin) {
            if self.in_snapshot() {
                self.undo_log.push(AddConstraint(constraint));
            }
        }
    }

    pub fn make_subregion(&mut self,
                          origin: SubregionOrigin,
                          sub: Region,
                          sup: Region) {
        // cannot add constraints once regions are resolved
        assert!(self.values.is_none());

        debug!("RegionVarBindings: make_subregion({:?}, {:?})", sub, sup);
        match (sub, sup) {
          (ReEarlyBound(..), _) |
          (ReLateBound(..), _) |
          (_, ReEarlyBound(..)) |
          (_, ReLateBound(..)) => {
            self.tcx.sess.span_bug(
                origin.span(),
                format!("Cannot relate bound region: {} <= {}",
                        sub.repr(self.tcx),
                        sup.repr(self.tcx)));
          }
          (ReInfer(ReVar(sub_id)), ReInfer(ReVar(sup_id))) => {
            self.add_constraint(ConstrainVarSubVar(sub_id, sup_id), origin);
          }
          (r, ReInfer(ReVar(sup_id))) => {
            self.add_constraint(ConstrainRegSubVar(r, sup_id), origin);
          }
          (ReInfer(ReVar(sub_id)), r) => {
            self.add_constraint(ConstrainVarSubReg(sub_id, r), origin);
          }
          _ => {
            self.add_constraint(ConstrainRegSubReg(sub, sup), origin);
          }
        }
    }

    pub fn lub_regions(&mut self,
                       origin: SubregionOrigin,
                       a: Region,
                       b: Region)
                       -> Region {
        // cannot add constraints once regions are resolved
        assert!(self.values.is_none());

        debug!("RegionVarBindings: lub_regions({:?}, {:?})", a, b);
        match (a, b) {
            (ReStatic, _) | (_, ReStatic) => {
                ReStatic // nothing lives longer than static
            }

            _ => {
                self.combine_vars(
                    Lub, a, b, origin,
                    |this, old_r, new_r|
                    this.make_subregion(origin, old_r, new_r))
            }
        }
    }

    pub fn glb_regions(&mut self,
                       origin: SubregionOrigin,
                       a: Region,
                       b: Region)
                       -> Region {
        // cannot add constraints once regions are resolved
        assert!(self.values.is_none());

        debug!("RegionVarBindings: glb_regions({:?}, {:?})", a, b);
        match (a, b) {
            (ReStatic, r) | (r, ReStatic) => {
                // static lives longer than everything else
                r
            }

            _ => {
                self.combine_vars(
                    Glb, a, b, origin,
                    |this, old_r, new_r|
                    this.make_subregion(origin, new_r, old_r))
            }
        }
    }

    pub fn resolve_var(&mut self, rid: RegionVid) -> ty::Region {
        let v = match self.values {
            None => self.tcx.sess.span_bug(
                self.var_origins[rid.to_uint()].span(),
                format!("Attempt to resolve region variable before values have \
                      been computed!")),
            Some(ref values) => values[rid.to_uint()]
        };

        debug!("RegionVarBindings: resolve_var({:?}={})={:?}",
               rid, rid.to_uint(), v);
        match v {
            Value(r) => r,

            NoValue => {
                // No constraints, return ty::ReEmpty
                ReEmpty
            }

            ErrorValue => {
                // An error that has previously been reported.
                ReStatic
            }
        }
    }

    fn combine_map<'a>(&'a mut self,
                       t: CombineMapType)
                       -> &'a mut CombineMap
    {
        match t {
            Glb => &mut self.glbs,
            Lub => &mut self.lubs,
        }
    }

    pub fn combine_vars(&mut self,
                        t: CombineMapType,
                        a: Region,
                        b: Region,
                        origin: SubregionOrigin,
                        relate: |this: &mut RegionVarBindings,
                                 old_r: Region,
                                 new_r: Region|)
                        -> Region {
        let vars = TwoRegions { a: a, b: b };
        match self.combine_map(t).find(&vars) {
            Some(&c) => {
                return ReInfer(ReVar(c));
            }
            None => {}
        }
        let c = self.new_region_var(infer::MiscVariable(origin.span()));
        self.combine_map(t).insert(vars, c);
        if self.in_snapshot() {
            self.undo_log.push(AddCombination(t, vars));
        }
        relate(self, a, ReInfer(ReVar(c)));
        relate(self, b, ReInfer(ReVar(c)));
        debug!("combine_vars() c={:?}", c);
        ReInfer(ReVar(c))
    }

    pub fn vars_created_since_snapshot(&mut self, snapshot: uint)
                                       -> ~[RegionVid] {
        self.undo_log.slice_from(snapshot).iter()
            .filter_map(|&elt| match elt {
                AddVar(vid) => Some(vid),
                _ => None
            })
            .collect()
    }

    pub fn tainted(&mut self, snapshot: uint, r0: Region) -> ~[Region] {
        /*!
         * Computes all regions that have been related to `r0` in any
         * way since the snapshot `snapshot` was taken---`r0` itself
         * will be the first entry. This is used when checking whether
         * skolemized regions are being improperly related to other
         * regions.
         */

        debug!("tainted(snapshot={}, r0={:?})", snapshot, r0);
        let _indenter = indenter();

        let undo_len = self.undo_log.len();

        // `result_set` acts as a worklist: we explore all outgoing
        // edges and add any new regions we find to result_set.  This
        // is not a terribly efficient implementation.
        let mut result_set = ~[r0];
        let mut result_index = 0;
        while result_index < result_set.len() {
            // nb: can't use uint::range() here because result_set grows
            let r = result_set[result_index];

            debug!("result_index={}, r={:?}", result_index, r);

            let mut undo_index = snapshot;
            while undo_index < undo_len {
                // nb: can't use uint::range() here as we move result_set
                let regs = match self.undo_log[undo_index] {
                    AddConstraint(ConstrainVarSubVar(ref a, ref b)) => {
                        Some((ReInfer(ReVar(*a)),
                              ReInfer(ReVar(*b))))
                    }
                    AddConstraint(ConstrainRegSubVar(ref a, ref b)) => {
                        Some((*a, ReInfer(ReVar(*b))))
                    }
                    AddConstraint(ConstrainVarSubReg(ref a, ref b)) => {
                        Some((ReInfer(ReVar(*a)), *b))
                    }
                    AddConstraint(ConstrainRegSubReg(a, b)) => {
                        Some((a, b))
                    }
                    _ => {
                        None
                    }
                };

                match regs {
                    None => {}
                    Some((r1, r2)) => {
                        result_set =
                            consider_adding_edge(result_set, r, r1, r2);
                        result_set =
                            consider_adding_edge(result_set, r, r2, r1);
                    }
                }

                undo_index += 1;
            }

            result_index += 1;
        }

        return result_set;

        fn consider_adding_edge(result_set: ~[Region],
                                r: Region,
                                r1: Region,
                                r2: Region) -> ~[Region]
        {
            let mut result_set = result_set;
            if r == r1 { // Clearly, this is potentially inefficient.
                if !result_set.iter().any(|x| *x == r2) {
                    result_set.push(r2);
                }
            }
            return result_set;
        }
    }

    /**
    This function performs the actual region resolution.  It must be
    called after all constraints have been added.  It performs a
    fixed-point iteration to find region values which satisfy all
    constraints, assuming such values can be found; if they cannot,
    errors are reported.
    */
    pub fn resolve_regions(&mut self) -> OptVec<RegionResolutionError> {
        debug!("RegionVarBindings: resolve_regions()");
        let mut errors = opt_vec::Empty;
        let v = self.infer_variable_values(&mut errors);
        self.values = Some(v);
        errors
    }
}

impl RegionVarBindings {
    fn is_subregion_of(&self, sub: Region, sup: Region) -> bool {
        let rm = self.tcx.region_maps;
        rm.is_subregion_of(sub, sup)
    }

    fn lub_concrete_regions(&self, a: Region, b: Region) -> Region {
        match (a, b) {
          (ReLateBound(..), _) |
          (_, ReLateBound(..)) |
          (ReEarlyBound(..), _) |
          (_, ReEarlyBound(..)) => {
            self.tcx.sess.bug(
                format!("Cannot relate bound region: LUB({}, {})",
                        a.repr(self.tcx),
                        b.repr(self.tcx)));
          }

          (ReStatic, _) | (_, ReStatic) => {
            ReStatic // nothing lives longer than static
          }

          (ReEmpty, r) | (r, ReEmpty) => {
            r // everything lives longer than empty
          }

          (ReInfer(ReVar(v_id)), _) | (_, ReInfer(ReVar(v_id))) => {
            self.tcx.sess.span_bug(
                self.var_origins[v_id.to_uint()].span(),
                format!("lub_concrete_regions invoked with \
                      non-concrete regions: {:?}, {:?}", a, b));
          }

          (f @ ReFree(ref fr), ReScope(s_id)) |
          (ReScope(s_id), f @ ReFree(ref fr)) => {
            // A "free" region can be interpreted as "some region
            // at least as big as the block fr.scope_id".  So, we can
            // reasonably compare free regions and scopes:
            let rm = self.tcx.region_maps;
            match rm.nearest_common_ancestor(fr.scope_id, s_id) {
              // if the free region's scope `fr.scope_id` is bigger than
              // the scope region `s_id`, then the LUB is the free
              // region itself:
              Some(r_id) if r_id == fr.scope_id => f,

              // otherwise, we don't know what the free region is,
              // so we must conservatively say the LUB is static:
              _ => ReStatic
            }
          }

          (ReScope(a_id), ReScope(b_id)) => {
            // The region corresponding to an outer block is a
            // subtype of the region corresponding to an inner
            // block.
            let rm = self.tcx.region_maps;
            match rm.nearest_common_ancestor(a_id, b_id) {
              Some(r_id) => ReScope(r_id),
              _ => ReStatic
            }
          }

          (ReFree(ref a_fr), ReFree(ref b_fr)) => {
             self.lub_free_regions(a_fr, b_fr)
          }

          // For these types, we cannot define any additional
          // relationship:
          (ReInfer(ReSkolemized(..)), _) |
          (_, ReInfer(ReSkolemized(..))) => {
            if a == b {a} else {ReStatic}
          }
        }
    }

    fn lub_free_regions(&self,
                        a: &FreeRegion,
                        b: &FreeRegion) -> ty::Region
    {
        /*!
         * Computes a region that encloses both free region arguments.
         * Guarantee that if the same two regions are given as argument,
         * in any order, a consistent result is returned.
         */

        return match a.cmp(b) {
            Less => helper(self, a, b),
            Greater => helper(self, b, a),
            Equal => ty::ReFree(*a)
        };

        fn helper(this: &RegionVarBindings,
                  a: &FreeRegion,
                  b: &FreeRegion) -> ty::Region
        {
            let rm = this.tcx.region_maps;
            if rm.sub_free_region(*a, *b) {
                ty::ReFree(*b)
            } else if rm.sub_free_region(*b, *a) {
                ty::ReFree(*a)
            } else {
                ty::ReStatic
            }
        }
    }

    fn glb_concrete_regions(&self,
                            a: Region,
                            b: Region)
                         -> cres<Region> {
        debug!("glb_concrete_regions({:?}, {:?})", a, b);
        match (a, b) {
            (ReLateBound(..), _) |
            (_, ReLateBound(..)) |
            (ReEarlyBound(..), _) |
            (_, ReEarlyBound(..)) => {
              self.tcx.sess.bug(
                  format!("Cannot relate bound region: GLB({}, {})",
                          a.repr(self.tcx),
                          b.repr(self.tcx)));
            }

            (ReStatic, r) | (r, ReStatic) => {
                // static lives longer than everything else
                Ok(r)
            }

            (ReEmpty, _) | (_, ReEmpty) => {
                // nothing lives shorter than everything else
                Ok(ReEmpty)
            }

            (ReInfer(ReVar(v_id)), _) |
            (_, ReInfer(ReVar(v_id))) => {
                self.tcx.sess.span_bug(
                    self.var_origins[v_id.to_uint()].span(),
                    format!("glb_concrete_regions invoked with \
                          non-concrete regions: {:?}, {:?}", a, b));
            }

            (ReFree(ref fr), s @ ReScope(s_id)) |
            (s @ ReScope(s_id), ReFree(ref fr)) => {
                // Free region is something "at least as big as
                // `fr.scope_id`."  If we find that the scope `fr.scope_id` is bigger
                // than the scope `s_id`, then we can say that the GLB
                // is the scope `s_id`.  Otherwise, as we do not know
                // big the free region is precisely, the GLB is undefined.
                let rm = self.tcx.region_maps;
                match rm.nearest_common_ancestor(fr.scope_id, s_id) {
                    Some(r_id) if r_id == fr.scope_id => Ok(s),
                    _ => Err(ty::terr_regions_no_overlap(b, a))
                }
            }

            (ReScope(a_id), ReScope(b_id)) => {
                self.intersect_scopes(a, b, a_id, b_id)
            }

            (ReFree(ref a_fr), ReFree(ref b_fr)) => {
                self.glb_free_regions(a_fr, b_fr)
            }

            // For these types, we cannot define any additional
            // relationship:
            (ReInfer(ReSkolemized(..)), _) |
            (_, ReInfer(ReSkolemized(..))) => {
                if a == b {
                    Ok(a)
                } else {
                    Err(ty::terr_regions_no_overlap(b, a))
                }
            }
        }
    }

    fn glb_free_regions(&self,
                        a: &FreeRegion,
                        b: &FreeRegion) -> cres<ty::Region>
    {
        /*!
         * Computes a region that is enclosed by both free region arguments,
         * if any. Guarantees that if the same two regions are given as argument,
         * in any order, a consistent result is returned.
         */

        return match a.cmp(b) {
            Less => helper(self, a, b),
            Greater => helper(self, b, a),
            Equal => Ok(ty::ReFree(*a))
        };

        fn helper(this: &RegionVarBindings,
                  a: &FreeRegion,
                  b: &FreeRegion) -> cres<ty::Region>
        {
            let rm = this.tcx.region_maps;
            if rm.sub_free_region(*a, *b) {
                Ok(ty::ReFree(*a))
            } else if rm.sub_free_region(*b, *a) {
                Ok(ty::ReFree(*b))
            } else {
                this.intersect_scopes(ty::ReFree(*a), ty::ReFree(*b),
                                      a.scope_id, b.scope_id)
            }
        }
    }

    fn report_type_error(&mut self,
                         origin: SubregionOrigin,
                         terr: &ty::type_err) {
        let terr_str = ty::type_err_to_str(self.tcx, terr);
        self.tcx.sess.span_err(origin.span(), terr_str);
    }

    fn intersect_scopes(&self,
                        region_a: ty::Region,
                        region_b: ty::Region,
                        scope_a: ast::NodeId,
                        scope_b: ast::NodeId) -> cres<Region>
    {
        // We want to generate the intersection of two
        // scopes or two free regions.  So, if one of
        // these scopes is a subscope of the other, return
        // it.  Otherwise fail.
        debug!("intersect_scopes(scope_a={:?}, scope_b={:?}, region_a={:?}, region_b={:?})",
               scope_a, scope_b, region_a, region_b);
        let rm = self.tcx.region_maps;
        match rm.nearest_common_ancestor(scope_a, scope_b) {
            Some(r_id) if scope_a == r_id => Ok(ReScope(scope_b)),
            Some(r_id) if scope_b == r_id => Ok(ReScope(scope_a)),
            _ => Err(ty::terr_regions_no_overlap(region_a, region_b))
        }
    }
}

// ______________________________________________________________________

#[deriving(Eq)]
enum Classification { Expanding, Contracting }

enum VarValue { NoValue, Value(Region), ErrorValue }

struct VarData {
    classification: Classification,
    value: VarValue,
}

struct RegionAndOrigin {
    region: Region,
    origin: SubregionOrigin,
}

type RegionGraph = graph::Graph<(), Constraint>;

impl RegionVarBindings {
    fn infer_variable_values(&self,
                             errors: &mut OptVec<RegionResolutionError>)
                             -> ~[VarValue] {
        let mut var_data = self.construct_var_data();
        self.expansion(var_data);
        self.contraction(var_data);
        self.collect_concrete_region_errors(errors);
        self.extract_values_and_collect_conflicts(var_data, errors)
    }

    fn construct_var_data(&self) -> ~[VarData] {
        vec::from_fn(self.num_vars(), |_| {
            VarData {
                // All nodes are initially classified as contracting; during
                // the expansion phase, we will shift the classification for
                // those nodes that have a concrete region predecessor to
                // Expanding.
                classification: Contracting,
                value: NoValue,
            }
        })
    }

    fn expansion(&self, var_data: &mut [VarData]) {
        self.iterate_until_fixed_point("Expansion", |constraint| {
            match *constraint {
              ConstrainRegSubVar(a_region, b_vid) => {
                let b_data = &mut var_data[b_vid.to_uint()];
                self.expand_node(a_region, b_vid, b_data)
              }
              ConstrainVarSubVar(a_vid, b_vid) => {
                match var_data[a_vid.to_uint()].value {
                  NoValue | ErrorValue => false,
                  Value(a_region) => {
                    let b_node = &mut var_data[b_vid.to_uint()];
                    self.expand_node(a_region, b_vid, b_node)
                  }
                }
              }
              ConstrainVarSubReg(..) => {
                // This is a contraction constraint.  Ignore it.
                false
              }
              ConstrainRegSubReg(..) => {
                // No region variables involved. Ignore.
                false
              }
            }
        })
    }

    fn expand_node(&self,
                   a_region: Region,
                   b_vid: RegionVid,
                   b_data: &mut VarData)
                   -> bool {
        debug!("expand_node({:?}, {:?} == {:?})",
               a_region, b_vid, b_data.value);

        b_data.classification = Expanding;
        match b_data.value {
          NoValue => {
            debug!("Setting initial value of {:?} to {:?}", b_vid, a_region);

            b_data.value = Value(a_region);
            return true;
          }

          Value(cur_region) => {
            let lub = self.lub_concrete_regions(a_region, cur_region);
            if lub == cur_region {
                return false;
            }

            debug!("Expanding value of {:?} from {:?} to {:?}",
                   b_vid, cur_region, lub);

            b_data.value = Value(lub);
            return true;
          }

          ErrorValue => {
            return false;
          }
        }
    }

    fn contraction(&self,
                   var_data: &mut [VarData]) {
        self.iterate_until_fixed_point("Contraction", |constraint| {
            match *constraint {
              ConstrainRegSubVar(..) => {
                // This is an expansion constraint.  Ignore.
                false
              }
              ConstrainVarSubVar(a_vid, b_vid) => {
                match var_data[b_vid.to_uint()].value {
                  NoValue | ErrorValue => false,
                  Value(b_region) => {
                    let a_data = &mut var_data[a_vid.to_uint()];
                    self.contract_node(a_vid, a_data, b_region)
                  }
                }
              }
              ConstrainVarSubReg(a_vid, b_region) => {
                let a_data = &mut var_data[a_vid.to_uint()];
                self.contract_node(a_vid, a_data, b_region)
              }
              ConstrainRegSubReg(..) => {
                // No region variables involved. Ignore.
                false
              }
            }
        })
    }

    fn contract_node(&self,
                     a_vid: RegionVid,
                     a_data: &mut VarData,
                     b_region: Region)
                     -> bool {
        debug!("contract_node({:?} == {:?}/{:?}, {:?})",
               a_vid, a_data.value, a_data.classification, b_region);

        return match a_data.value {
            NoValue => {
                assert_eq!(a_data.classification, Contracting);
                a_data.value = Value(b_region);
                true // changed
            }

            ErrorValue => {
                false // no change
            }

            Value(a_region) => {
                match a_data.classification {
                    Expanding => {
                        check_node(self, a_vid, a_data, a_region, b_region)
                    }
                    Contracting => {
                        adjust_node(self, a_vid, a_data, a_region, b_region)
                    }
                }
            }
        };

        fn check_node(this: &RegionVarBindings,
                      a_vid: RegionVid,
                      a_data: &mut VarData,
                      a_region: Region,
                      b_region: Region)
                   -> bool {
            if !this.is_subregion_of(a_region, b_region) {
                debug!("Setting {:?} to ErrorValue: {:?} not subregion of {:?}",
                       a_vid, a_region, b_region);
                a_data.value = ErrorValue;
            }
            false
        }

        fn adjust_node(this: &RegionVarBindings,
                       a_vid: RegionVid,
                       a_data: &mut VarData,
                       a_region: Region,
                       b_region: Region)
                    -> bool {
            match this.glb_concrete_regions(a_region, b_region) {
                Ok(glb) => {
                    if glb == a_region {
                        false
                    } else {
                        debug!("Contracting value of {:?} from {:?} to {:?}",
                               a_vid, a_region, glb);
                        a_data.value = Value(glb);
                        true
                    }
                }
                Err(_) => {
                    debug!("Setting {:?} to ErrorValue: no glb of {:?}, {:?}",
                           a_vid, a_region, b_region);
                    a_data.value = ErrorValue;
                    false
                }
            }
        }
    }

    fn collect_concrete_region_errors(
        &self,
        errors: &mut OptVec<RegionResolutionError>)
    {
        for (constraint, _) in self.constraints.iter() {
            let (sub, sup) = match *constraint {
                ConstrainVarSubVar(..) |
                ConstrainRegSubVar(..) |
                ConstrainVarSubReg(..) => {
                    continue;
                }
                ConstrainRegSubReg(sub, sup) => {
                    (sub, sup)
                }
            };

            if self.is_subregion_of(sub, sup) {
                continue;
            }

            debug!("ConcreteFailure: !(sub <= sup): sub={:?}, sup={:?}",
                   sub, sup);
            let origin = self.constraints.get_copy(constraint);
            errors.push(ConcreteFailure(origin, sub, sup));
        }
    }

    fn extract_values_and_collect_conflicts(
        &self,
        var_data: &[VarData],
        errors: &mut OptVec<RegionResolutionError>)
        -> ~[VarValue]
    {
        debug!("extract_values_and_collect_conflicts()");

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
        let mut dup_vec = vec::from_elem(self.num_vars(), uint::max_value);

        let mut opt_graph = None;

        for idx in range(0u, self.num_vars()) {
            match var_data[idx].value {
                Value(_) => {
                    /* Inference successful */
                }
                NoValue => {
                    /* Unconstrained inference: do not report an error
                       until the value of this variable is requested.
                       After all, sometimes we make region variables but never
                       really use their values. */
                }
                ErrorValue => {
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

                    if opt_graph.is_none() {
                        opt_graph = Some(self.construct_graph());
                    }
                    let graph = opt_graph.get_ref();

                    let node_vid = RegionVid { id: idx };
                    match var_data[idx].classification {
                        Expanding => {
                            self.collect_error_for_expanding_node(
                                graph, var_data, dup_vec, node_vid, errors);
                        }
                        Contracting => {
                            self.collect_error_for_contracting_node(
                                graph, var_data, dup_vec, node_vid, errors);
                        }
                    }
                }
            }
        }

        vec::from_fn(self.num_vars(), |idx| var_data[idx].value)
    }

    fn construct_graph(&self) -> RegionGraph {
        let num_vars = self.num_vars();
        let num_edges = self.constraints.len();

        let mut graph = graph::Graph::with_capacity(num_vars + 1,
                                                    num_edges);

        for _ in range(0u, num_vars) {
            graph.add_node(());
        }
        let dummy_idx = graph.add_node(());

        for (constraint, _) in self.constraints.iter() {
            match *constraint {
                ConstrainVarSubVar(a_id, b_id) => {
                    graph.add_edge(NodeIndex(a_id.to_uint()),
                                   NodeIndex(b_id.to_uint()),
                                   *constraint);
                }
                ConstrainRegSubVar(_, b_id) => {
                    graph.add_edge(dummy_idx,
                                   NodeIndex(b_id.to_uint()),
                                   *constraint);
                }
                ConstrainVarSubReg(a_id, _) => {
                    graph.add_edge(NodeIndex(a_id.to_uint()),
                                   dummy_idx,
                                   *constraint);
                }
                ConstrainRegSubReg(..) => {
                    // Relations between two concrete regions do not
                    // require an edge in the graph.
                }
            }
        }

        return graph;
    }

    fn collect_error_for_expanding_node(
        &self,
        graph: &RegionGraph,
        var_data: &[VarData],
        dup_vec: &mut [uint],
        node_idx: RegionVid,
        errors: &mut OptVec<RegionResolutionError>)
    {
        // Errors in expanding nodes result from a lower-bound that is
        // not contained by an upper-bound.
        let (lower_bounds, lower_dup) =
            self.collect_concrete_regions(graph, var_data, node_idx,
                                          graph::Incoming, dup_vec);
        let (upper_bounds, upper_dup) =
            self.collect_concrete_regions(graph, var_data, node_idx,
                                          graph::Outgoing, dup_vec);

        if lower_dup || upper_dup {
            return;
        }

        for lower_bound in lower_bounds.iter() {
            for upper_bound in upper_bounds.iter() {
                if !self.is_subregion_of(lower_bound.region,
                                         upper_bound.region) {
                    errors.push(SubSupConflict(
                        self.var_origins[node_idx.to_uint()],
                        lower_bound.origin,
                        lower_bound.region,
                        upper_bound.origin,
                        upper_bound.region));
                    return;
                }
            }
        }

        self.tcx.sess.span_bug(
            self.var_origins[node_idx.to_uint()].span(),
            format!("collect_error_for_expanding_node() could not find error \
                  for var {:?}, lower_bounds={}, upper_bounds={}",
                 node_idx,
                 lower_bounds.map(|x| x.region).repr(self.tcx),
                 upper_bounds.map(|x| x.region).repr(self.tcx)));
    }

    fn collect_error_for_contracting_node(
        &self,
        graph: &RegionGraph,
        var_data: &[VarData],
        dup_vec: &mut [uint],
        node_idx: RegionVid,
        errors: &mut OptVec<RegionResolutionError>)
    {
        // Errors in contracting nodes result from two upper-bounds
        // that have no intersection.
        let (upper_bounds, dup_found) =
            self.collect_concrete_regions(graph, var_data, node_idx,
                                          graph::Outgoing, dup_vec);

        if dup_found {
            return;
        }

        for upper_bound_1 in upper_bounds.iter() {
            for upper_bound_2 in upper_bounds.iter() {
                match self.glb_concrete_regions(upper_bound_1.region,
                                                upper_bound_2.region) {
                  Ok(_) => {}
                  Err(_) => {
                    errors.push(SupSupConflict(
                        self.var_origins[node_idx.to_uint()],
                        upper_bound_1.origin,
                        upper_bound_1.region,
                        upper_bound_2.origin,
                        upper_bound_2.region));
                    return;
                  }
                }
            }
        }

        self.tcx.sess.span_bug(
            self.var_origins[node_idx.to_uint()].span(),
            format!("collect_error_for_contracting_node() could not find error \
                  for var {:?}, upper_bounds={}",
                 node_idx,
                 upper_bounds.map(|x| x.region).repr(self.tcx)));
    }

    fn collect_concrete_regions(&self,
                                graph: &RegionGraph,
                                var_data: &[VarData],
                                orig_node_idx: RegionVid,
                                dir: Direction,
                                dup_vec: &mut [uint])
                                -> (~[RegionAndOrigin], bool) {
        struct WalkState {
            set: HashSet<RegionVid>,
            stack: ~[RegionVid],
            result: ~[RegionAndOrigin],
            dup_found: bool
        }
        let mut state = WalkState {
            set: HashSet::new(),
            stack: ~[orig_node_idx],
            result: ~[],
            dup_found: false
        };
        state.set.insert(orig_node_idx);

        // to start off the process, walk the source node in the
        // direction specified
        process_edges(self, &mut state, graph, orig_node_idx, dir);

        while !state.stack.is_empty() {
            let node_idx = state.stack.pop();
            let classification = var_data[node_idx.to_uint()].classification;

            // check whether we've visited this node on some previous walk
            if dup_vec[node_idx.to_uint()] == uint::max_value {
                dup_vec[node_idx.to_uint()] = orig_node_idx.to_uint();
            } else if dup_vec[node_idx.to_uint()] != orig_node_idx.to_uint() {
                state.dup_found = true;
            }

            debug!("collect_concrete_regions(orig_node_idx={:?}, node_idx={:?}, \
                    classification={:?})",
                   orig_node_idx, node_idx, classification);

            // figure out the direction from which this node takes its
            // values, and search for concrete regions etc in that direction
            let dir = match classification {
                Expanding => graph::Incoming,
                Contracting => graph::Outgoing,
            };

            process_edges(self, &mut state, graph, node_idx, dir);
        }

        let WalkState {result, dup_found, ..} = state;
        return (result, dup_found);

        fn process_edges(this: &RegionVarBindings,
                         state: &mut WalkState,
                         graph: &RegionGraph,
                         source_vid: RegionVid,
                         dir: Direction) {
            debug!("process_edges(source_vid={:?}, dir={:?})", source_vid, dir);

            let source_node_index = NodeIndex(source_vid.to_uint());
            graph.each_adjacent_edge(source_node_index, dir, |_, edge| {
                match edge.data {
                    ConstrainVarSubVar(from_vid, to_vid) => {
                        let opp_vid =
                            if from_vid == source_vid {to_vid} else {from_vid};
                        if state.set.insert(opp_vid) {
                            state.stack.push(opp_vid);
                        }
                    }

                    ConstrainRegSubVar(region, _) |
                    ConstrainVarSubReg(_, region) => {
                        state.result.push(RegionAndOrigin {
                            region: region,
                            origin: this.constraints.get_copy(&edge.data)
                        });
                    }

                    ConstrainRegSubReg(..) => {}
                }
                true
            });
        }
    }

    fn iterate_until_fixed_point(&self,
                                 tag: &str,
                                 body: |constraint: &Constraint| -> bool) {
        let mut iteration = 0;
        let mut changed = true;
        while changed {
            changed = false;
            iteration += 1;
            debug!("---- {} Iteration \\#{}", tag, iteration);
            for (constraint, _) in self.constraints.iter() {
                let edge_changed = body(constraint);
                if edge_changed {
                    debug!("Updated due to constraint {}",
                           constraint.repr(self.tcx));
                    changed = true;
                }
            }
        }
        debug!("---- {} Complete after {} iteration(s)", tag, iteration);
    }

}

impl Repr for Constraint {
    fn repr(&self, tcx: ty::ctxt) -> ~str {
        match *self {
            ConstrainVarSubVar(a, b) => format!("ConstrainVarSubVar({}, {})",
                                             a.repr(tcx), b.repr(tcx)),
            ConstrainRegSubVar(a, b) => format!("ConstrainRegSubVar({}, {})",
                                             a.repr(tcx), b.repr(tcx)),
            ConstrainVarSubReg(a, b) => format!("ConstrainVarSubReg({}, {})",
                                             a.repr(tcx), b.repr(tcx)),
            ConstrainRegSubReg(a, b) => format!("ConstrainRegSubReg({}, {})",
                                             a.repr(tcx), b.repr(tcx)),
        }
    }
}
