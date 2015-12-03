// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! See README.md

pub use self::Constraint::*;
pub use self::Verify::*;
pub use self::UndoLogEntry::*;
pub use self::CombineMapType::*;
pub use self::RegionResolutionError::*;
pub use self::VarValue::*;

use super::{RegionVariableOrigin, SubregionOrigin, TypeTrace, MiscVariable};

use rustc_data_structures::graph::{self, Direction, NodeIndex};
use middle::free_region::FreeRegionMap;
use middle::ty::{self, Ty};
use middle::ty::{BoundRegion, FreeRegion, Region, RegionVid};
use middle::ty::{ReEmpty, ReStatic, ReFree, ReEarlyBound};
use middle::ty::{ReLateBound, ReScope, ReVar, ReSkolemized, BrFresh};
use middle::ty::error::TypeError;
use util::common::indenter;
use util::nodemap::{FnvHashMap, FnvHashSet};

use std::cell::{Cell, RefCell};
use std::cmp::Ordering::{self, Less, Greater, Equal};
use std::fmt;
use std::u32;
use syntax::ast;

mod graphviz;

// A constraint that influences the inference process.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Constraint {
    // One region variable is subregion of another
    ConstrainVarSubVar(RegionVid, RegionVid),

    // Concrete region is subregion of region variable
    ConstrainRegSubVar(Region, RegionVid),

    // Region variable is subregion of concrete region
    //
    // FIXME(#29436) -- should be remove in favor of a Verify
    ConstrainVarSubReg(RegionVid, Region),
}

// Something we have to verify after region inference is done, but
// which does not directly influence the inference process
pub enum Verify<'tcx> {
    // VerifyRegSubReg(a, b): Verify that `a <= b`. Neither `a` nor
    // `b` are inference variables.
    VerifyRegSubReg(SubregionOrigin<'tcx>, Region, Region),

    // VerifyGenericBound(T, _, R, RS): The parameter type `T` (or
    // associated type) must outlive the region `R`. `T` is known to
    // outlive `RS`. Therefore verify that `R <= RS[i]` for some
    // `i`. Inference variables may be involved (but this verification
    // step doesn't influence inference).
    VerifyGenericBound(GenericKind<'tcx>, SubregionOrigin<'tcx>, Region, VerifyBound),
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum GenericKind<'tcx> {
    Param(ty::ParamTy),
    Projection(ty::ProjectionTy<'tcx>),
}

// When we introduce a verification step, we wish to test that a
// particular region (let's call it `'min`) meets some bound.
// The bound is described the by the following grammar:
#[derive(Debug)]
pub enum VerifyBound {
    // B = exists {R} --> some 'r in {R} must outlive 'min
    //
    // Put another way, the subject value is known to outlive all
    // regions in {R}, so if any of those outlives 'min, then the
    // bound is met.
    AnyRegion(Vec<Region>),

    // B = forall {R} --> all 'r in {R} must outlive 'min
    //
    // Put another way, the subject value is known to outlive some
    // region in {R}, so if all of those outlives 'min, then the bound
    // is met.
    AllRegions(Vec<Region>),

    // B = exists {B} --> 'min must meet some bound b in {B}
    AnyBound(Vec<VerifyBound>),

    // B = forall {B} --> 'min must meet all bounds b in {B}
    AllBounds(Vec<VerifyBound>),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct TwoRegions {
    a: Region,
    b: Region,
}

#[derive(Copy, Clone, PartialEq)]
pub enum UndoLogEntry {
    OpenSnapshot,
    CommitedSnapshot,
    AddVar(RegionVid),
    AddConstraint(Constraint),
    AddVerify(usize),
    AddGiven(ty::FreeRegion, ty::RegionVid),
    AddCombination(CombineMapType, TwoRegions),
}

#[derive(Copy, Clone, PartialEq)]
pub enum CombineMapType {
    Lub,
    Glb,
}

#[derive(Clone, Debug)]
pub enum RegionResolutionError<'tcx> {
    /// `ConcreteFailure(o, a, b)`:
    ///
    /// `o` requires that `a <= b`, but this does not hold
    ConcreteFailure(SubregionOrigin<'tcx>, Region, Region),

    /// `GenericBoundFailure(p, s, a)
    ///
    /// The parameter/associated-type `p` must be known to outlive the lifetime
    /// `a` (but none of the known bounds are sufficient).
    GenericBoundFailure(SubregionOrigin<'tcx>, GenericKind<'tcx>, Region),

    /// `SubSupConflict(v, sub_origin, sub_r, sup_origin, sup_r)`:
    ///
    /// Could not infer a value for `v` because `sub_r <= v` (due to
    /// `sub_origin`) but `v <= sup_r` (due to `sup_origin`) and
    /// `sub_r <= sup_r` does not hold.
    SubSupConflict(RegionVariableOrigin,
                   SubregionOrigin<'tcx>,
                   Region,
                   SubregionOrigin<'tcx>,
                   Region),

    /// For subsets of `ConcreteFailure` and `SubSupConflict`, we can derive
    /// more specific errors message by suggesting to the user where they
    /// should put a lifetime. In those cases we process and put those errors
    /// into `ProcessedErrors` before we do any reporting.
    ProcessedErrors(Vec<RegionVariableOrigin>,
                    Vec<(TypeTrace<'tcx>, TypeError<'tcx>)>,
                    Vec<SameRegions>),
}

/// SameRegions is used to group regions that we think are the same and would
/// like to indicate so to the user.
/// For example, the following function
/// ```
/// struct Foo { bar: i32 }
/// fn foo2<'a, 'b>(x: &'a Foo) -> &'b i32 {
///    &x.bar
/// }
/// ```
/// would report an error because we expect 'a and 'b to match, and so we group
/// 'a and 'b together inside a SameRegions struct
#[derive(Clone, Debug)]
pub struct SameRegions {
    pub scope_id: ast::NodeId,
    pub regions: Vec<BoundRegion>,
}

impl SameRegions {
    pub fn contains(&self, other: &BoundRegion) -> bool {
        self.regions.contains(other)
    }

    pub fn push(&mut self, other: BoundRegion) {
        self.regions.push(other);
    }
}

pub type CombineMap = FnvHashMap<TwoRegions, RegionVid>;

pub struct RegionVarBindings<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    var_origins: RefCell<Vec<RegionVariableOrigin>>,

    // Constraints of the form `A <= B` introduced by the region
    // checker.  Here at least one of `A` and `B` must be a region
    // variable.
    constraints: RefCell<FnvHashMap<Constraint, SubregionOrigin<'tcx>>>,

    // A "verify" is something that we need to verify after inference is
    // done, but which does not directly affect inference in any way.
    //
    // An example is a `A <= B` where neither `A` nor `B` are
    // inference variables.
    verifys: RefCell<Vec<Verify<'tcx>>>,

    // A "given" is a relationship that is known to hold. In particular,
    // we often know from closure fn signatures that a particular free
    // region must be a subregion of a region variable:
    //
    //    foo.iter().filter(<'a> |x: &'a &'b T| ...)
    //
    // In situations like this, `'b` is in fact a region variable
    // introduced by the call to `iter()`, and `'a` is a bound region
    // on the closure (as indicated by the `<'a>` prefix). If we are
    // naive, we wind up inferring that `'b` must be `'static`,
    // because we require that it be greater than `'a` and we do not
    // know what `'a` is precisely.
    //
    // This hashmap is used to avoid that naive scenario. Basically we
    // record the fact that `'a <= 'b` is implied by the fn signature,
    // and then ignore the constraint when solving equations. This is
    // a bit of a hack but seems to work.
    givens: RefCell<FnvHashSet<(ty::FreeRegion, ty::RegionVid)>>,

    lubs: RefCell<CombineMap>,
    glbs: RefCell<CombineMap>,
    skolemization_count: Cell<u32>,
    bound_count: Cell<u32>,

    // The undo log records actions that might later be undone.
    //
    // Note: when the undo_log is empty, we are not actively
    // snapshotting. When the `start_snapshot()` method is called, we
    // push an OpenSnapshot entry onto the list to indicate that we
    // are now actively snapshotting. The reason for this is that
    // otherwise we end up adding entries for things like the lower
    // bound on a variable and so forth, which can never be rolled
    // back.
    undo_log: RefCell<Vec<UndoLogEntry>>,

    // This contains the results of inference.  It begins as an empty
    // option and only acquires a value after inference is complete.
    values: RefCell<Option<Vec<VarValue>>>,
}

#[derive(Debug)]
pub struct RegionSnapshot {
    length: usize,
    skolemization_count: u32,
}

impl<'a, 'tcx> RegionVarBindings<'a, 'tcx> {
    pub fn new(tcx: &'a ty::ctxt<'tcx>) -> RegionVarBindings<'a, 'tcx> {
        RegionVarBindings {
            tcx: tcx,
            var_origins: RefCell::new(Vec::new()),
            values: RefCell::new(None),
            constraints: RefCell::new(FnvHashMap()),
            verifys: RefCell::new(Vec::new()),
            givens: RefCell::new(FnvHashSet()),
            lubs: RefCell::new(FnvHashMap()),
            glbs: RefCell::new(FnvHashMap()),
            skolemization_count: Cell::new(0),
            bound_count: Cell::new(0),
            undo_log: RefCell::new(Vec::new()),
        }
    }

    fn in_snapshot(&self) -> bool {
        !self.undo_log.borrow().is_empty()
    }

    pub fn start_snapshot(&self) -> RegionSnapshot {
        let length = self.undo_log.borrow().len();
        debug!("RegionVarBindings: start_snapshot({})", length);
        self.undo_log.borrow_mut().push(OpenSnapshot);
        RegionSnapshot {
            length: length,
            skolemization_count: self.skolemization_count.get(),
        }
    }

    pub fn commit(&self, snapshot: RegionSnapshot) {
        debug!("RegionVarBindings: commit({})", snapshot.length);
        assert!(self.undo_log.borrow().len() > snapshot.length);
        assert!((*self.undo_log.borrow())[snapshot.length] == OpenSnapshot);

        let mut undo_log = self.undo_log.borrow_mut();
        if snapshot.length == 0 {
            undo_log.truncate(0);
        } else {
            (*undo_log)[snapshot.length] = CommitedSnapshot;
        }
        self.skolemization_count.set(snapshot.skolemization_count);
    }

    pub fn rollback_to(&self, snapshot: RegionSnapshot) {
        debug!("RegionVarBindings: rollback_to({:?})", snapshot);
        let mut undo_log = self.undo_log.borrow_mut();
        assert!(undo_log.len() > snapshot.length);
        assert!((*undo_log)[snapshot.length] == OpenSnapshot);
        while undo_log.len() > snapshot.length + 1 {
            match undo_log.pop().unwrap() {
                OpenSnapshot => {
                    panic!("Failure to observe stack discipline");
                }
                CommitedSnapshot => {}
                AddVar(vid) => {
                    let mut var_origins = self.var_origins.borrow_mut();
                    var_origins.pop().unwrap();
                    assert_eq!(var_origins.len(), vid.index as usize);
                }
                AddConstraint(ref constraint) => {
                    self.constraints.borrow_mut().remove(constraint);
                }
                AddVerify(index) => {
                    self.verifys.borrow_mut().pop();
                    assert_eq!(self.verifys.borrow().len(), index);
                }
                AddGiven(sub, sup) => {
                    self.givens.borrow_mut().remove(&(sub, sup));
                }
                AddCombination(Glb, ref regions) => {
                    self.glbs.borrow_mut().remove(regions);
                }
                AddCombination(Lub, ref regions) => {
                    self.lubs.borrow_mut().remove(regions);
                }
            }
        }
        let c = undo_log.pop().unwrap();
        assert!(c == OpenSnapshot);
        self.skolemization_count.set(snapshot.skolemization_count);
    }

    pub fn num_vars(&self) -> u32 {
        let len = self.var_origins.borrow().len();
        // enforce no overflow
        assert!(len as u32 as usize == len);
        len as u32
    }

    pub fn new_region_var(&self, origin: RegionVariableOrigin) -> RegionVid {
        let id = self.num_vars();
        self.var_origins.borrow_mut().push(origin.clone());
        let vid = RegionVid { index: id };
        if self.in_snapshot() {
            self.undo_log.borrow_mut().push(AddVar(vid));
        }
        debug!("created new region variable {:?} with origin {:?}",
               vid,
               origin);
        return vid;
    }

    /// Creates a new skolemized region. Skolemized regions are fresh
    /// regions used when performing higher-ranked computations. They
    /// must be used in a very particular way and are never supposed
    /// to "escape" out into error messages or the code at large.
    ///
    /// The idea is to always create a snapshot. Skolemized regions
    /// can be created in the context of this snapshot, but once the
    /// snapshot is committed or rolled back, their numbers will be
    /// recycled, so you must be finished with them. See the extensive
    /// comments in `higher_ranked.rs` to see how it works (in
    /// particular, the subtyping comparison).
    ///
    /// The `snapshot` argument to this function is not really used;
    /// it's just there to make it explicit which snapshot bounds the
    /// skolemized region that results.
    pub fn new_skolemized(&self, br: ty::BoundRegion, snapshot: &RegionSnapshot) -> Region {
        assert!(self.in_snapshot());
        assert!(self.undo_log.borrow()[snapshot.length] == OpenSnapshot);

        let sc = self.skolemization_count.get();
        self.skolemization_count.set(sc + 1);
        ReSkolemized(ty::SkolemizedRegionVid { index: sc }, br)
    }

    pub fn new_bound(&self, debruijn: ty::DebruijnIndex) -> Region {
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

        let sc = self.bound_count.get();
        self.bound_count.set(sc + 1);

        if sc >= self.bound_count.get() {
            self.tcx.sess.bug("rollover in RegionInference new_bound()");
        }

        ReLateBound(debruijn, BrFresh(sc))
    }

    fn values_are_none(&self) -> bool {
        self.values.borrow().is_none()
    }

    fn add_constraint(&self, constraint: Constraint, origin: SubregionOrigin<'tcx>) {
        // cannot add constraints once regions are resolved
        assert!(self.values_are_none());

        debug!("RegionVarBindings: add_constraint({:?})", constraint);

        if self.constraints.borrow_mut().insert(constraint, origin).is_none() {
            if self.in_snapshot() {
                self.undo_log.borrow_mut().push(AddConstraint(constraint));
            }
        }
    }

    fn add_verify(&self, verify: Verify<'tcx>) {
        // cannot add verifys once regions are resolved
        assert!(self.values_are_none());

        debug!("RegionVarBindings: add_verify({:?})", verify);

        // skip no-op cases known to be satisfied
        match verify {
            VerifyGenericBound(_, _, _, VerifyBound::AllBounds(ref bs)) if bs.len() == 0 => {
                return;
            }
            _ => {}
        }

        let mut verifys = self.verifys.borrow_mut();
        let index = verifys.len();
        verifys.push(verify);
        if self.in_snapshot() {
            self.undo_log.borrow_mut().push(AddVerify(index));
        }
    }

    pub fn add_given(&self, sub: ty::FreeRegion, sup: ty::RegionVid) {
        // cannot add givens once regions are resolved
        assert!(self.values_are_none());

        let mut givens = self.givens.borrow_mut();
        if givens.insert((sub, sup)) {
            debug!("add_given({:?} <= {:?})", sub, sup);

            self.undo_log.borrow_mut().push(AddGiven(sub, sup));
        }
    }

    pub fn make_eqregion(&self, origin: SubregionOrigin<'tcx>, sub: Region, sup: Region) {
        if sub != sup {
            // Eventually, it would be nice to add direct support for
            // equating regions.
            self.make_subregion(origin.clone(), sub, sup);
            self.make_subregion(origin, sup, sub);
        }
    }

    pub fn make_subregion(&self, origin: SubregionOrigin<'tcx>, sub: Region, sup: Region) {
        // cannot add constraints once regions are resolved
        assert!(self.values_are_none());

        debug!("RegionVarBindings: make_subregion({:?}, {:?}) due to {:?}",
               sub,
               sup,
               origin);

        match (sub, sup) {
            (ReEarlyBound(..), ReEarlyBound(..)) => {
                // This case is used only to make sure that explicitly-specified
                // `Self` types match the real self type in implementations.
                //
                // FIXME(NDM) -- we really shouldn't be comparing bound things
                self.add_verify(VerifyRegSubReg(origin, sub, sup));
            }
            (ReEarlyBound(..), _) |
            (ReLateBound(..), _) |
            (_, ReEarlyBound(..)) |
            (_, ReLateBound(..)) => {
                self.tcx.sess.span_bug(origin.span(),
                                       &format!("cannot relate bound region: {:?} <= {:?}",
                                                sub,
                                                sup));
            }
            (_, ReStatic) => {
                // all regions are subregions of static, so we can ignore this
            }
            (ReVar(sub_id), ReVar(sup_id)) => {
                self.add_constraint(ConstrainVarSubVar(sub_id, sup_id), origin);
            }
            (r, ReVar(sup_id)) => {
                self.add_constraint(ConstrainRegSubVar(r, sup_id), origin);
            }
            (ReVar(sub_id), r) => {
                self.add_constraint(ConstrainVarSubReg(sub_id, r), origin);
            }
            _ => {
                self.add_verify(VerifyRegSubReg(origin, sub, sup));
            }
        }
    }

    /// See `Verify::VerifyGenericBound`
    pub fn verify_generic_bound(&self,
                                origin: SubregionOrigin<'tcx>,
                                kind: GenericKind<'tcx>,
                                sub: Region,
                                bound: VerifyBound) {
        self.add_verify(VerifyGenericBound(kind, origin, sub, bound));
    }

    pub fn lub_regions(&self, origin: SubregionOrigin<'tcx>, a: Region, b: Region) -> Region {
        // cannot add constraints once regions are resolved
        assert!(self.values_are_none());

        debug!("RegionVarBindings: lub_regions({:?}, {:?})", a, b);
        match (a, b) {
            (ReStatic, _) | (_, ReStatic) => {
                ReStatic // nothing lives longer than static
            }

            _ => {
                self.combine_vars(Lub, a, b, origin.clone(), |this, old_r, new_r| {
                    this.make_subregion(origin.clone(), old_r, new_r)
                })
            }
        }
    }

    pub fn glb_regions(&self, origin: SubregionOrigin<'tcx>, a: Region, b: Region) -> Region {
        // cannot add constraints once regions are resolved
        assert!(self.values_are_none());

        debug!("RegionVarBindings: glb_regions({:?}, {:?})", a, b);
        match (a, b) {
            (ReStatic, r) | (r, ReStatic) => {
                // static lives longer than everything else
                r
            }

            _ => {
                self.combine_vars(Glb, a, b, origin.clone(), |this, old_r, new_r| {
                    this.make_subregion(origin.clone(), new_r, old_r)
                })
            }
        }
    }

    pub fn resolve_var(&self, rid: RegionVid) -> ty::Region {
        match *self.values.borrow() {
            None => {
                self.tcx.sess.span_bug((*self.var_origins.borrow())[rid.index as usize].span(),
                                       "attempt to resolve region variable before values have \
                                        been computed!")
            }
            Some(ref values) => {
                let r = lookup(values, rid);
                debug!("resolve_var({:?}) = {:?}", rid, r);
                r
            }
        }
    }

    fn combine_map(&self, t: CombineMapType) -> &RefCell<CombineMap> {
        match t {
            Glb => &self.glbs,
            Lub => &self.lubs,
        }
    }

    pub fn combine_vars<F>(&self,
                           t: CombineMapType,
                           a: Region,
                           b: Region,
                           origin: SubregionOrigin<'tcx>,
                           mut relate: F)
                           -> Region
        where F: FnMut(&RegionVarBindings<'a, 'tcx>, Region, Region)
    {
        let vars = TwoRegions { a: a, b: b };
        match self.combine_map(t).borrow().get(&vars) {
            Some(&c) => {
                return ReVar(c);
            }
            None => {}
        }
        let c = self.new_region_var(MiscVariable(origin.span()));
        self.combine_map(t).borrow_mut().insert(vars, c);
        if self.in_snapshot() {
            self.undo_log.borrow_mut().push(AddCombination(t, vars));
        }
        relate(self, a, ReVar(c));
        relate(self, b, ReVar(c));
        debug!("combine_vars() c={:?}", c);
        ReVar(c)
    }

    pub fn vars_created_since_snapshot(&self, mark: &RegionSnapshot) -> Vec<RegionVid> {
        self.undo_log.borrow()[mark.length..]
            .iter()
            .filter_map(|&elt| {
                match elt {
                    AddVar(vid) => Some(vid),
                    _ => None,
                }
            })
            .collect()
    }

    /// Computes all regions that have been related to `r0` in any way since the mark `mark` was
    /// made---`r0` itself will be the first entry. This is used when checking whether skolemized
    /// regions are being improperly related to other regions.
    pub fn tainted(&self, mark: &RegionSnapshot, r0: Region) -> Vec<Region> {
        debug!("tainted(mark={:?}, r0={:?})", mark, r0);
        let _indenter = indenter();

        // `result_set` acts as a worklist: we explore all outgoing
        // edges and add any new regions we find to result_set.  This
        // is not a terribly efficient implementation.
        let mut result_set = vec![r0];
        let mut result_index = 0;
        while result_index < result_set.len() {
            // nb: can't use usize::range() here because result_set grows
            let r = result_set[result_index];
            debug!("result_index={}, r={:?}", result_index, r);

            for undo_entry in self.undo_log.borrow()[mark.length..].iter() {
                match undo_entry {
                    &AddConstraint(ConstrainVarSubVar(a, b)) => {
                        consider_adding_bidirectional_edges(&mut result_set, r, ReVar(a), ReVar(b));
                    }
                    &AddConstraint(ConstrainRegSubVar(a, b)) => {
                        consider_adding_bidirectional_edges(&mut result_set, r, a, ReVar(b));
                    }
                    &AddConstraint(ConstrainVarSubReg(a, b)) => {
                        consider_adding_bidirectional_edges(&mut result_set, r, ReVar(a), b);
                    }
                    &AddGiven(a, b) => {
                        consider_adding_bidirectional_edges(&mut result_set,
                                                            r,
                                                            ReFree(a),
                                                            ReVar(b));
                    }
                    &AddVerify(i) => {
                        match (*self.verifys.borrow())[i] {
                            VerifyRegSubReg(_, a, b) => {
                                consider_adding_bidirectional_edges(&mut result_set, r, a, b);
                            }
                            VerifyGenericBound(_, _, a, ref bound) => {
                                bound.for_each_region(&mut |b| {
                                    consider_adding_bidirectional_edges(&mut result_set, r, a, b)
                                });
                            }
                        }
                    }
                    &AddCombination(..) |
                    &AddVar(..) |
                    &OpenSnapshot |
                    &CommitedSnapshot => {}
                }
            }

            result_index += 1;
        }

        return result_set;

        fn consider_adding_bidirectional_edges(result_set: &mut Vec<Region>,
                                               r: Region,
                                               r1: Region,
                                               r2: Region) {
            consider_adding_directed_edge(result_set, r, r1, r2);
            consider_adding_directed_edge(result_set, r, r2, r1);
        }

        fn consider_adding_directed_edge(result_set: &mut Vec<Region>,
                                         r: Region,
                                         r1: Region,
                                         r2: Region) {
            if r == r1 {
                // Clearly, this is potentially inefficient.
                if !result_set.iter().any(|x| *x == r2) {
                    result_set.push(r2);
                }
            }
        }
    }

    /// This function performs the actual region resolution.  It must be
    /// called after all constraints have been added.  It performs a
    /// fixed-point iteration to find region values which satisfy all
    /// constraints, assuming such values can be found; if they cannot,
    /// errors are reported.
    pub fn resolve_regions(&self,
                           free_regions: &FreeRegionMap,
                           subject_node: ast::NodeId)
                           -> Vec<RegionResolutionError<'tcx>> {
        debug!("RegionVarBindings: resolve_regions()");
        let mut errors = vec![];
        let v = self.infer_variable_values(free_regions, &mut errors, subject_node);
        *self.values.borrow_mut() = Some(v);
        errors
    }

    fn lub_concrete_regions(&self, free_regions: &FreeRegionMap, a: Region, b: Region) -> Region {
        match (a, b) {
            (ReLateBound(..), _) |
            (_, ReLateBound(..)) |
            (ReEarlyBound(..), _) |
            (_, ReEarlyBound(..)) => {
                self.tcx.sess.bug(&format!("cannot relate bound region: LUB({:?}, {:?})", a, b));
            }

            (ReStatic, _) | (_, ReStatic) => {
                ReStatic // nothing lives longer than static
            }

            (ReEmpty, r) | (r, ReEmpty) => {
                r // everything lives longer than empty
            }

            (ReVar(v_id), _) | (_, ReVar(v_id)) => {
                self.tcx.sess.span_bug((*self.var_origins.borrow())[v_id.index as usize].span(),
                                       &format!("lub_concrete_regions invoked with non-concrete \
                                                 regions: {:?}, {:?}",
                                                a,
                                                b));
            }

            (ReFree(ref fr), ReScope(s_id)) |
            (ReScope(s_id), ReFree(ref fr)) => {
                let f = ReFree(*fr);
                // A "free" region can be interpreted as "some region
                // at least as big as the block fr.scope_id".  So, we can
                // reasonably compare free regions and scopes:
                let r_id = self.tcx.region_maps.nearest_common_ancestor(fr.scope, s_id);

                if r_id == fr.scope {
                    // if the free region's scope `fr.scope_id` is bigger than
                    // the scope region `s_id`, then the LUB is the free
                    // region itself:
                    f
                } else {
                    // otherwise, we don't know what the free region is,
                    // so we must conservatively say the LUB is static:
                    ReStatic
                }
            }

            (ReScope(a_id), ReScope(b_id)) => {
                // The region corresponding to an outer block is a
                // subtype of the region corresponding to an inner
                // block.
                ReScope(self.tcx.region_maps.nearest_common_ancestor(a_id, b_id))
            }

            (ReFree(a_fr), ReFree(b_fr)) => {
                free_regions.lub_free_regions(a_fr, b_fr)
            }

            // For these types, we cannot define any additional
            // relationship:
            (ReSkolemized(..), _) |
            (_, ReSkolemized(..)) => {
                if a == b {
                    a
                } else {
                    ReStatic
                }
            }
        }
    }
}

// ______________________________________________________________________

#[derive(Copy, Clone, Debug)]
pub enum VarValue {
    Value(Region),
    ErrorValue,
}

struct VarData {
    value: VarValue,
}

struct RegionAndOrigin<'tcx> {
    region: Region,
    origin: SubregionOrigin<'tcx>,
}

type RegionGraph = graph::Graph<(), Constraint>;

impl<'a, 'tcx> RegionVarBindings<'a, 'tcx> {
    fn infer_variable_values(&self,
                             free_regions: &FreeRegionMap,
                             errors: &mut Vec<RegionResolutionError<'tcx>>,
                             subject: ast::NodeId)
                             -> Vec<VarValue> {
        let mut var_data = self.construct_var_data();

        // Dorky hack to cause `dump_constraints` to only get called
        // if debug mode is enabled:
        debug!("----() End constraint listing (subject={}) {:?}---",
               subject,
               self.dump_constraints(subject));
        graphviz::maybe_print_constraints_for(self, subject);

        let graph = self.construct_graph();
        self.expand_givens(&graph);
        self.expansion(free_regions, &mut var_data);
        self.contraction(free_regions, &mut var_data);
        let values = self.extract_values_and_collect_conflicts(free_regions,
                                                               &var_data,
                                                               &graph,
                                                               errors);
        self.collect_concrete_region_errors(free_regions, &values, errors);
        values
    }

    fn construct_var_data(&self) -> Vec<VarData> {
        (0..self.num_vars() as usize)
            .map(|_| VarData { value: Value(ty::ReEmpty) })
            .collect()
    }

    fn dump_constraints(&self, subject: ast::NodeId) {
        debug!("----() Start constraint listing (subject={}) ()----",
               subject);
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
        for (fr, vid) in seeds {
            let seed_index = NodeIndex(vid.index as usize);
            for succ_index in graph.depth_traverse(seed_index) {
                let succ_index = succ_index.0 as u32;
                if succ_index < self.num_vars() {
                    let succ_vid = RegionVid { index: succ_index };
                    givens.insert((fr, succ_vid));
                }
            }
        }
    }

    fn expansion(&self, free_regions: &FreeRegionMap, var_data: &mut [VarData]) {
        self.iterate_until_fixed_point("Expansion", |constraint| {
            debug!("expansion: constraint={:?} origin={:?}",
                   constraint,
                   self.constraints
                       .borrow()
                       .get(constraint)
                       .unwrap());
            match *constraint {
                ConstrainRegSubVar(a_region, b_vid) => {
                    let b_data = &mut var_data[b_vid.index as usize];
                    self.expand_node(free_regions, a_region, b_vid, b_data)
                }
                ConstrainVarSubVar(a_vid, b_vid) => {
                    match var_data[a_vid.index as usize].value {
                        ErrorValue => false,
                        Value(a_region) => {
                            let b_node = &mut var_data[b_vid.index as usize];
                            self.expand_node(free_regions, a_region, b_vid, b_node)
                        }
                    }
                }
                ConstrainVarSubReg(..) => {
                    // This is a contraction constraint.  Ignore it.
                    false
                }
            }
        })
    }

    fn expand_node(&self,
                   free_regions: &FreeRegionMap,
                   a_region: Region,
                   b_vid: RegionVid,
                   b_data: &mut VarData)
                   -> bool {
        debug!("expand_node({:?}, {:?} == {:?})",
               a_region,
               b_vid,
               b_data.value);

        // Check if this relationship is implied by a given.
        match a_region {
            ty::ReFree(fr) => {
                if self.givens.borrow().contains(&(fr, b_vid)) {
                    debug!("given");
                    return false;
                }
            }
            _ => {}
        }

        match b_data.value {
            Value(cur_region) => {
                let lub = self.lub_concrete_regions(free_regions, a_region, cur_region);
                if lub == cur_region {
                    return false;
                }

                debug!("Expanding value of {:?} from {:?} to {:?}",
                       b_vid,
                       cur_region,
                       lub);

                b_data.value = Value(lub);
                return true;
            }

            ErrorValue => {
                return false;
            }
        }
    }

    // FIXME(#29436) -- this fn would just go away if we removed ConstrainVarSubReg
    fn contraction(&self, free_regions: &FreeRegionMap, var_data: &mut [VarData]) {
        self.iterate_until_fixed_point("Contraction", |constraint| {
            debug!("contraction: constraint={:?} origin={:?}",
                   constraint,
                   self.constraints
                       .borrow()
                       .get(constraint)
                       .unwrap());
            match *constraint {
                ConstrainRegSubVar(..) |
                ConstrainVarSubVar(..) => {
                    // Expansion will ensure that these constraints hold. Ignore.
                }
                ConstrainVarSubReg(a_vid, b_region) => {
                    let a_data = &mut var_data[a_vid.index as usize];
                    debug!("contraction: {:?} == {:?}, {:?}",
                           a_vid,
                           a_data.value,
                           b_region);

                    let a_region = match a_data.value {
                        ErrorValue => return false,
                        Value(a_region) => a_region,
                    };

                    if !free_regions.is_subregion_of(self.tcx, a_region, b_region) {
                        debug!("Setting {:?} to ErrorValue: {:?} not subregion of {:?}",
                               a_vid,
                               a_region,
                               b_region);
                        a_data.value = ErrorValue;
                    }
                }
            }

            false
        })
    }

    fn collect_concrete_region_errors(&self,
                                      free_regions: &FreeRegionMap,
                                      values: &Vec<VarValue>,
                                      errors: &mut Vec<RegionResolutionError<'tcx>>) {
        let mut reg_reg_dups = FnvHashSet();
        for verify in self.verifys.borrow().iter() {
            match *verify {
                VerifyRegSubReg(ref origin, sub, sup) => {
                    if free_regions.is_subregion_of(self.tcx, sub, sup) {
                        continue;
                    }

                    if !reg_reg_dups.insert((sub, sup)) {
                        continue;
                    }

                    debug!("region inference error at {:?}: {:?} <= {:?} is not true",
                           origin,
                           sub,
                           sup);

                    errors.push(ConcreteFailure((*origin).clone(), sub, sup));
                }

                VerifyGenericBound(ref kind, ref origin, sub, ref bound) => {
                    let sub = normalize(values, sub);
                    if bound.is_met(self.tcx, free_regions, values, sub) {
                        continue;
                    }

                    debug!("region inference error at {:?}: verifying {:?} <= {:?}",
                           origin,
                           sub,
                           bound);

                    errors.push(GenericBoundFailure((*origin).clone(), kind.clone(), sub));
                }
            }
        }
    }

    fn extract_values_and_collect_conflicts(&self,
                                            free_regions: &FreeRegionMap,
                                            var_data: &[VarData],
                                            graph: &RegionGraph,
                                            errors: &mut Vec<RegionResolutionError<'tcx>>)
                                            -> Vec<VarValue> {
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
        let mut dup_vec = vec![u32::MAX; self.num_vars() as usize];

        for idx in 0..self.num_vars() as usize {
            match var_data[idx].value {
                Value(_) => {
                    /* Inference successful */
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

                    let node_vid = RegionVid { index: idx as u32 };
                    self.collect_error_for_expanding_node(free_regions,
                                                          graph,
                                                          &mut dup_vec,
                                                          node_vid,
                                                          errors);
                }
            }
        }

        (0..self.num_vars() as usize).map(|idx| var_data[idx].value).collect()
    }

    fn construct_graph(&self) -> RegionGraph {
        let num_vars = self.num_vars();

        let constraints = self.constraints.borrow();

        let mut graph = graph::Graph::new();

        for _ in 0..num_vars {
            graph.add_node(());
        }
        let dummy_idx = graph.add_node(());

        for (constraint, _) in constraints.iter() {
            match *constraint {
                ConstrainVarSubVar(a_id, b_id) => {
                    graph.add_edge(NodeIndex(a_id.index as usize),
                                   NodeIndex(b_id.index as usize),
                                   *constraint);
                }
                ConstrainRegSubVar(_, b_id) => {
                    graph.add_edge(dummy_idx, NodeIndex(b_id.index as usize), *constraint);
                }
                ConstrainVarSubReg(a_id, _) => {
                    graph.add_edge(NodeIndex(a_id.index as usize), dummy_idx, *constraint);
                }
            }
        }

        return graph;
    }

    fn collect_error_for_expanding_node(&self,
                                        free_regions: &FreeRegionMap,
                                        graph: &RegionGraph,
                                        dup_vec: &mut [u32],
                                        node_idx: RegionVid,
                                        errors: &mut Vec<RegionResolutionError<'tcx>>) {
        // Errors in expanding nodes result from a lower-bound that is
        // not contained by an upper-bound.
        let (mut lower_bounds, lower_dup) = self.collect_concrete_regions(graph,
                                                                          node_idx,
                                                                          graph::INCOMING,
                                                                          dup_vec);
        let (mut upper_bounds, upper_dup) = self.collect_concrete_regions(graph,
                                                                          node_idx,
                                                                          graph::OUTGOING,
                                                                          dup_vec);

        if lower_dup || upper_dup {
            return;
        }

        // We place free regions first because we are special casing
        // SubSupConflict(ReFree, ReFree) when reporting error, and so
        // the user will more likely get a specific suggestion.
        fn free_regions_first(a: &RegionAndOrigin, b: &RegionAndOrigin) -> Ordering {
            match (a.region, b.region) {
                (ReFree(..), ReFree(..)) => Equal,
                (ReFree(..), _) => Less,
                (_, ReFree(..)) => Greater,
                (_, _) => Equal,
            }
        }
        lower_bounds.sort_by(|a, b| free_regions_first(a, b));
        upper_bounds.sort_by(|a, b| free_regions_first(a, b));

        for lower_bound in &lower_bounds {
            for upper_bound in &upper_bounds {
                if !free_regions.is_subregion_of(self.tcx, lower_bound.region, upper_bound.region) {
                    let origin = (*self.var_origins.borrow())[node_idx.index as usize].clone();
                    debug!("region inference error at {:?} for {:?}: SubSupConflict sub: {:?} \
                            sup: {:?}",
                           origin,
                           node_idx,
                           lower_bound.region,
                           upper_bound.region);
                    errors.push(SubSupConflict(origin,
                                               lower_bound.origin.clone(),
                                               lower_bound.region,
                                               upper_bound.origin.clone(),
                                               upper_bound.region));
                    return;
                }
            }
        }

        self.tcx.sess.span_bug((*self.var_origins.borrow())[node_idx.index as usize].span(),
                               &format!("collect_error_for_expanding_node() could not find \
                                         error for var {:?}, lower_bounds={:?}, \
                                         upper_bounds={:?}",
                                        node_idx,
                                        lower_bounds,
                                        upper_bounds));
    }

    fn collect_concrete_regions(&self,
                                graph: &RegionGraph,
                                orig_node_idx: RegionVid,
                                dir: Direction,
                                dup_vec: &mut [u32])
                                -> (Vec<RegionAndOrigin<'tcx>>, bool) {
        struct WalkState<'tcx> {
            set: FnvHashSet<RegionVid>,
            stack: Vec<RegionVid>,
            result: Vec<RegionAndOrigin<'tcx>>,
            dup_found: bool,
        }
        let mut state = WalkState {
            set: FnvHashSet(),
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

            debug!("collect_concrete_regions(orig_node_idx={:?}, node_idx={:?})",
                   orig_node_idx,
                   node_idx);

            // figure out the direction from which this node takes its
            // values, and search for concrete regions etc in that direction
            let dir = graph::INCOMING;
            process_edges(self, &mut state, graph, node_idx, dir);
        }

        let WalkState {result, dup_found, ..} = state;
        return (result, dup_found);

        fn process_edges<'a, 'tcx>(this: &RegionVarBindings<'a, 'tcx>,
                                   state: &mut WalkState<'tcx>,
                                   graph: &RegionGraph,
                                   source_vid: RegionVid,
                                   dir: Direction) {
            debug!("process_edges(source_vid={:?}, dir={:?})", source_vid, dir);

            let source_node_index = NodeIndex(source_vid.index as usize);
            for (_, edge) in graph.adjacent_edges(source_node_index, dir) {
                match edge.data {
                    ConstrainVarSubVar(from_vid, to_vid) => {
                        let opp_vid = if from_vid == source_vid {
                            to_vid
                        } else {
                            from_vid
                        };
                        if state.set.insert(opp_vid) {
                            state.stack.push(opp_vid);
                        }
                    }

                    ConstrainRegSubVar(region, _) |
                    ConstrainVarSubReg(_, region) => {
                        state.result.push(RegionAndOrigin {
                            region: region,
                            origin: this.constraints.borrow().get(&edge.data).unwrap().clone(),
                        });
                    }
                }
            }
        }
    }

    fn iterate_until_fixed_point<F>(&self, tag: &str, mut body: F)
        where F: FnMut(&Constraint) -> bool
    {
        let mut iteration = 0;
        let mut changed = true;
        while changed {
            changed = false;
            iteration += 1;
            debug!("---- {} Iteration {}{}", "#", tag, iteration);
            for (constraint, _) in self.constraints.borrow().iter() {
                let edge_changed = body(constraint);
                if edge_changed {
                    debug!("Updated due to constraint {:?}", constraint);
                    changed = true;
                }
            }
        }
        debug!("---- {} Complete after {} iteration(s)", tag, iteration);
    }

}

impl<'tcx> fmt::Debug for Verify<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            VerifyRegSubReg(_, ref a, ref b) => {
                write!(f, "VerifyRegSubReg({:?}, {:?})", a, b)
            }
            VerifyGenericBound(_, ref p, ref a, ref bs) => {
                write!(f, "VerifyGenericBound({:?}, {:?}, {:?})", p, a, bs)
            }
        }
    }
}

fn normalize(values: &Vec<VarValue>, r: ty::Region) -> ty::Region {
    match r {
        ty::ReVar(rid) => lookup(values, rid),
        _ => r,
    }
}

fn lookup(values: &Vec<VarValue>, rid: ty::RegionVid) -> ty::Region {
    match values[rid.index as usize] {
        Value(r) => r,
        ErrorValue => ReStatic, // Previously reported error.
    }
}

impl<'tcx> fmt::Debug for RegionAndOrigin<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RegionAndOrigin({:?},{:?})", self.region, self.origin)
    }
}

impl<'tcx> fmt::Debug for GenericKind<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            GenericKind::Param(ref p) => write!(f, "{:?}", p),
            GenericKind::Projection(ref p) => write!(f, "{:?}", p),
        }
    }
}

impl<'tcx> fmt::Display for GenericKind<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            GenericKind::Param(ref p) => write!(f, "{}", p),
            GenericKind::Projection(ref p) => write!(f, "{}", p),
        }
    }
}

impl<'tcx> GenericKind<'tcx> {
    pub fn to_ty(&self, tcx: &ty::ctxt<'tcx>) -> Ty<'tcx> {
        match *self {
            GenericKind::Param(ref p) => p.to_ty(tcx),
            GenericKind::Projection(ref p) => tcx.mk_projection(p.trait_ref.clone(), p.item_name),
        }
    }
}

impl VerifyBound {
    fn for_each_region(&self, f: &mut FnMut(ty::Region)) {
        match self {
            &VerifyBound::AnyRegion(ref rs) |
            &VerifyBound::AllRegions(ref rs) => for &r in rs {
                f(r);
            },

            &VerifyBound::AnyBound(ref bs) |
            &VerifyBound::AllBounds(ref bs) => for b in bs {
                b.for_each_region(f);
            },
        }
    }

    pub fn must_hold(&self) -> bool {
        match self {
            &VerifyBound::AnyRegion(ref bs) => bs.contains(&ty::ReStatic),
            &VerifyBound::AllRegions(ref bs) => bs.is_empty(),
            &VerifyBound::AnyBound(ref bs) => bs.iter().any(|b| b.must_hold()),
            &VerifyBound::AllBounds(ref bs) => bs.iter().all(|b| b.must_hold()),
        }
    }

    pub fn cannot_hold(&self) -> bool {
        match self {
            &VerifyBound::AnyRegion(ref bs) => bs.is_empty(),
            &VerifyBound::AllRegions(ref bs) => bs.contains(&ty::ReEmpty),
            &VerifyBound::AnyBound(ref bs) => bs.iter().all(|b| b.cannot_hold()),
            &VerifyBound::AllBounds(ref bs) => bs.iter().any(|b| b.cannot_hold()),
        }
    }

    pub fn or(self, vb: VerifyBound) -> VerifyBound {
        if self.must_hold() || vb.cannot_hold() {
            self
        } else if self.cannot_hold() || vb.must_hold() {
            vb
        } else {
            VerifyBound::AnyBound(vec![self, vb])
        }
    }

    pub fn and(self, vb: VerifyBound) -> VerifyBound {
        if self.must_hold() && vb.must_hold() {
            self
        } else if self.cannot_hold() && vb.cannot_hold() {
            self
        } else {
            VerifyBound::AllBounds(vec![self, vb])
        }
    }

    fn is_met<'tcx>(&self,
                    tcx: &ty::ctxt<'tcx>,
                    free_regions: &FreeRegionMap,
                    var_values: &Vec<VarValue>,
                    min: ty::Region)
                    -> bool {
        match self {
            &VerifyBound::AnyRegion(ref rs) =>
                rs.iter()
                  .map(|&r| normalize(var_values, r))
                  .any(|r| free_regions.is_subregion_of(tcx, min, r)),

            &VerifyBound::AllRegions(ref rs) =>
                rs.iter()
                  .map(|&r| normalize(var_values, r))
                  .all(|r| free_regions.is_subregion_of(tcx, min, r)),

            &VerifyBound::AnyBound(ref bs) =>
                bs.iter()
                  .any(|b| b.is_met(tcx, free_regions, var_values, min)),

            &VerifyBound::AllBounds(ref bs) =>
                bs.iter()
                  .all(|b| b.is_met(tcx, free_regions, var_values, min)),
        }
    }
}
