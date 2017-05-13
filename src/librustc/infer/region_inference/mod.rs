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
pub use self::UndoLogEntry::*;
pub use self::CombineMapType::*;
pub use self::RegionResolutionError::*;
pub use self::VarValue::*;

use super::{RegionVariableOrigin, SubregionOrigin, MiscVariable};
use super::unify_key;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::graph::{self, Direction, NodeIndex, OUTGOING};
use rustc_data_structures::unify::{self, UnificationTable};
use middle::free_region::FreeRegionMap;
use ty::{self, Ty, TyCtxt};
use ty::{BoundRegion, Region, RegionVid};
use ty::{ReEmpty, ReStatic, ReFree, ReEarlyBound, ReErased};
use ty::{ReLateBound, ReScope, ReVar, ReSkolemized, BrFresh};

use std::cell::{Cell, RefCell};
use std::cmp::Ordering::{self, Less, Greater, Equal};
use std::fmt;
use std::mem;
use std::u32;
use syntax::ast;

mod graphviz;

// A constraint that influences the inference process.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Constraint<'tcx> {
    // One region variable is subregion of another
    ConstrainVarSubVar(RegionVid, RegionVid),

    // Concrete region is subregion of region variable
    ConstrainRegSubVar(&'tcx Region, RegionVid),

    // Region variable is subregion of concrete region. This does not
    // directly affect inference, but instead is checked after
    // inference is complete.
    ConstrainVarSubReg(RegionVid, &'tcx Region),

    // A constraint where neither side is a variable. This does not
    // directly affect inference, but instead is checked after
    // inference is complete.
    ConstrainRegSubReg(&'tcx Region, &'tcx Region),
}

// VerifyGenericBound(T, _, R, RS): The parameter type `T` (or
// associated type) must outlive the region `R`. `T` is known to
// outlive `RS`. Therefore verify that `R <= RS[i]` for some
// `i`. Inference variables may be involved (but this verification
// step doesn't influence inference).
#[derive(Debug)]
pub struct Verify<'tcx> {
    kind: GenericKind<'tcx>,
    origin: SubregionOrigin<'tcx>,
    region: &'tcx Region,
    bound: VerifyBound<'tcx>,
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
pub enum VerifyBound<'tcx> {
    // B = exists {R} --> some 'r in {R} must outlive 'min
    //
    // Put another way, the subject value is known to outlive all
    // regions in {R}, so if any of those outlives 'min, then the
    // bound is met.
    AnyRegion(Vec<&'tcx Region>),

    // B = forall {R} --> all 'r in {R} must outlive 'min
    //
    // Put another way, the subject value is known to outlive some
    // region in {R}, so if all of those outlives 'min, then the bound
    // is met.
    AllRegions(Vec<&'tcx Region>),

    // B = exists {B} --> 'min must meet some bound b in {B}
    AnyBound(Vec<VerifyBound<'tcx>>),

    // B = forall {B} --> 'min must meet all bounds b in {B}
    AllBounds(Vec<VerifyBound<'tcx>>),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct TwoRegions<'tcx> {
    a: &'tcx Region,
    b: &'tcx Region,
}

#[derive(Copy, Clone, PartialEq)]
pub enum UndoLogEntry<'tcx> {
    /// Pushed when we start a snapshot.
    OpenSnapshot,

    /// Replaces an `OpenSnapshot` when a snapshot is committed, but
    /// that snapshot is not the root. If the root snapshot is
    /// unrolled, all nested snapshots must be committed.
    CommitedSnapshot,

    /// We added `RegionVid`
    AddVar(RegionVid),

    /// We added the given `constraint`
    AddConstraint(Constraint<'tcx>),

    /// We added the given `verify`
    AddVerify(usize),

    /// We added the given `given`
    AddGiven(ty::FreeRegion, ty::RegionVid),

    /// We added a GLB/LUB "combinaton variable"
    AddCombination(CombineMapType, TwoRegions<'tcx>),

    /// During skolemization, we sometimes purge entries from the undo
    /// log in a kind of minisnapshot (unlike other snapshots, this
    /// purging actually takes place *on success*). In that case, we
    /// replace the corresponding entry with `Noop` so as to avoid the
    /// need to do a bunch of swapping. (We can't use `swap_remove` as
    /// the order of the vector is important.)
    Purged,
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
    ConcreteFailure(SubregionOrigin<'tcx>, &'tcx Region, &'tcx Region),

    /// `GenericBoundFailure(p, s, a)
    ///
    /// The parameter/associated-type `p` must be known to outlive the lifetime
    /// `a` (but none of the known bounds are sufficient).
    GenericBoundFailure(SubregionOrigin<'tcx>, GenericKind<'tcx>, &'tcx Region),

    /// `SubSupConflict(v, sub_origin, sub_r, sup_origin, sup_r)`:
    ///
    /// Could not infer a value for `v` because `sub_r <= v` (due to
    /// `sub_origin`) but `v <= sup_r` (due to `sup_origin`) and
    /// `sub_r <= sup_r` does not hold.
    SubSupConflict(RegionVariableOrigin,
                   SubregionOrigin<'tcx>,
                   &'tcx Region,
                   SubregionOrigin<'tcx>,
                   &'tcx Region),

    /// For subsets of `ConcreteFailure` and `SubSupConflict`, we can derive
    /// more specific errors message by suggesting to the user where they
    /// should put a lifetime. In those cases we process and put those errors
    /// into `ProcessedErrors` before we do any reporting.
    ProcessedErrors(Vec<ProcessedErrorOrigin<'tcx>>,
                    Vec<SameRegions>),
}

#[derive(Clone, Debug)]
pub enum ProcessedErrorOrigin<'tcx> {
    ConcreteFailure(SubregionOrigin<'tcx>, &'tcx Region, &'tcx Region),
    VariableFailure(RegionVariableOrigin),
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

pub type CombineMap<'tcx> = FxHashMap<TwoRegions<'tcx>, RegionVid>;

pub struct RegionVarBindings<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    var_origins: RefCell<Vec<RegionVariableOrigin>>,

    // Constraints of the form `A <= B` introduced by the region
    // checker.  Here at least one of `A` and `B` must be a region
    // variable.
    constraints: RefCell<FxHashMap<Constraint<'tcx>, SubregionOrigin<'tcx>>>,

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
    givens: RefCell<FxHashSet<(ty::FreeRegion, ty::RegionVid)>>,

    lubs: RefCell<CombineMap<'tcx>>,
    glbs: RefCell<CombineMap<'tcx>>,
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
    undo_log: RefCell<Vec<UndoLogEntry<'tcx>>>,
    unification_table: RefCell<UnificationTable<ty::RegionVid>>,

    // This contains the results of inference.  It begins as an empty
    // option and only acquires a value after inference is complete.
    values: RefCell<Option<Vec<VarValue<'tcx>>>>,
}

pub struct RegionSnapshot {
    length: usize,
    region_snapshot: unify::Snapshot<ty::RegionVid>,
    skolemization_count: u32,
}

/// When working with skolemized regions, we often wish to find all of
/// the regions that are either reachable from a skolemized region, or
/// which can reach a skolemized region, or both. We call such regions
/// *tained* regions.  This struct allows you to decide what set of
/// tainted regions you want.
#[derive(Debug)]
pub struct TaintDirections {
    incoming: bool,
    outgoing: bool,
}

impl TaintDirections {
    pub fn incoming() -> Self {
        TaintDirections { incoming: true, outgoing: false }
    }

    pub fn outgoing() -> Self {
        TaintDirections { incoming: false, outgoing: true }
    }

    pub fn both() -> Self {
        TaintDirections { incoming: true, outgoing: true }
    }
}

struct TaintSet<'tcx> {
    directions: TaintDirections,
    regions: FxHashSet<&'tcx ty::Region>
}

impl<'a, 'gcx, 'tcx> TaintSet<'tcx> {
    fn new(directions: TaintDirections,
           initial_region: &'tcx ty::Region)
           -> Self {
        let mut regions = FxHashSet();
        regions.insert(initial_region);
        TaintSet { directions: directions, regions: regions }
    }

    fn fixed_point(&mut self,
                   tcx: TyCtxt<'a, 'gcx, 'tcx>,
                   undo_log: &[UndoLogEntry<'tcx>],
                   verifys: &[Verify<'tcx>]) {
        let mut prev_len = 0;
        while prev_len < self.len() {
            debug!("tainted: prev_len = {:?} new_len = {:?}",
                   prev_len, self.len());

            prev_len = self.len();

            for undo_entry in undo_log {
                match undo_entry {
                    &AddConstraint(ConstrainVarSubVar(a, b)) => {
                        self.add_edge(tcx.mk_region(ReVar(a)),
                                      tcx.mk_region(ReVar(b)));
                    }
                    &AddConstraint(ConstrainRegSubVar(a, b)) => {
                        self.add_edge(a, tcx.mk_region(ReVar(b)));
                    }
                    &AddConstraint(ConstrainVarSubReg(a, b)) => {
                        self.add_edge(tcx.mk_region(ReVar(a)), b);
                    }
                    &AddConstraint(ConstrainRegSubReg(a, b)) => {
                        self.add_edge(a, b);
                    }
                    &AddGiven(a, b) => {
                        self.add_edge(tcx.mk_region(ReFree(a)),
                                      tcx.mk_region(ReVar(b)));
                    }
                    &AddVerify(i) => {
                        verifys[i].bound.for_each_region(&mut |b| {
                            self.add_edge(verifys[i].region, b);
                        });
                    }
                    &Purged |
                    &AddCombination(..) |
                    &AddVar(..) |
                    &OpenSnapshot |
                    &CommitedSnapshot => {}
                }
            }
        }
    }

    fn into_set(self) -> FxHashSet<&'tcx ty::Region> {
        self.regions
    }

    fn len(&self) -> usize {
        self.regions.len()
    }

    fn add_edge(&mut self,
                source: &'tcx ty::Region,
                target: &'tcx ty::Region) {
        if self.directions.incoming {
            if self.regions.contains(&target) {
                self.regions.insert(source);
            }
        }

        if self.directions.outgoing {
            if self.regions.contains(&source) {
                self.regions.insert(target);
            }
        }
    }
}

impl<'a, 'gcx, 'tcx> RegionVarBindings<'a, 'gcx, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'gcx, 'tcx>) -> RegionVarBindings<'a, 'gcx, 'tcx> {
        RegionVarBindings {
            tcx: tcx,
            var_origins: RefCell::new(Vec::new()),
            values: RefCell::new(None),
            constraints: RefCell::new(FxHashMap()),
            verifys: RefCell::new(Vec::new()),
            givens: RefCell::new(FxHashSet()),
            lubs: RefCell::new(FxHashMap()),
            glbs: RefCell::new(FxHashMap()),
            skolemization_count: Cell::new(0),
            bound_count: Cell::new(0),
            undo_log: RefCell::new(Vec::new()),
            unification_table: RefCell::new(UnificationTable::new()),
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
            region_snapshot: self.unification_table.borrow_mut().snapshot(),
            skolemization_count: self.skolemization_count.get(),
        }
    }

    pub fn commit(&self, snapshot: RegionSnapshot) {
        debug!("RegionVarBindings: commit({})", snapshot.length);
        assert!(self.undo_log.borrow().len() > snapshot.length);
        assert!((*self.undo_log.borrow())[snapshot.length] == OpenSnapshot);
        assert!(self.skolemization_count.get() == snapshot.skolemization_count,
                "failed to pop skolemized regions: {} now vs {} at start",
                self.skolemization_count.get(),
                snapshot.skolemization_count);

        let mut undo_log = self.undo_log.borrow_mut();
        if snapshot.length == 0 {
            undo_log.truncate(0);
        } else {
            (*undo_log)[snapshot.length] = CommitedSnapshot;
        }
        self.unification_table.borrow_mut().commit(snapshot.region_snapshot);
    }

    pub fn rollback_to(&self, snapshot: RegionSnapshot) {
        debug!("RegionVarBindings: rollback_to({:?})", snapshot);
        let mut undo_log = self.undo_log.borrow_mut();
        assert!(undo_log.len() > snapshot.length);
        assert!((*undo_log)[snapshot.length] == OpenSnapshot);
        while undo_log.len() > snapshot.length + 1 {
            self.rollback_undo_entry(undo_log.pop().unwrap());
        }
        let c = undo_log.pop().unwrap();
        assert!(c == OpenSnapshot);
        self.skolemization_count.set(snapshot.skolemization_count);
        self.unification_table.borrow_mut()
            .rollback_to(snapshot.region_snapshot);
    }

    pub fn rollback_undo_entry(&self, undo_entry: UndoLogEntry<'tcx>) {
        match undo_entry {
            OpenSnapshot => {
                panic!("Failure to observe stack discipline");
            }
            Purged | CommitedSnapshot => {
                // nothing to do here
            }
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

    pub fn num_vars(&self) -> u32 {
        let len = self.var_origins.borrow().len();
        // enforce no overflow
        assert!(len as u32 as usize == len);
        len as u32
    }

    pub fn new_region_var(&self, origin: RegionVariableOrigin) -> RegionVid {
        let vid = RegionVid { index: self.num_vars() };
        self.var_origins.borrow_mut().push(origin.clone());

        let u_vid = self.unification_table.borrow_mut().new_key(
            unify_key::RegionVidKey { min_vid: vid }
            );
        assert_eq!(vid, u_vid);
        if self.in_snapshot() {
            self.undo_log.borrow_mut().push(AddVar(vid));
        }
        debug!("created new region variable {:?} with origin {:?}",
               vid,
               origin);
        return vid;
    }

    pub fn var_origin(&self, vid: RegionVid) -> RegionVariableOrigin {
        self.var_origins.borrow()[vid.index as usize].clone()
    }

    /// Creates a new skolemized region. Skolemized regions are fresh
    /// regions used when performing higher-ranked computations. They
    /// must be used in a very particular way and are never supposed
    /// to "escape" out into error messages or the code at large.
    ///
    /// The idea is to always create a snapshot. Skolemized regions
    /// can be created in the context of this snapshot, but before the
    /// snapshot is committed or rolled back, they must be popped
    /// (using `pop_skolemized_regions`), so that their numbers can be
    /// recycled. Normally you don't have to think about this: you use
    /// the APIs in `higher_ranked/mod.rs`, such as
    /// `skolemize_late_bound_regions` and `plug_leaks`, which will
    /// guide you on this path (ensure that the `SkolemizationMap` is
    /// consumed and you are good).  There are also somewhat extensive
    /// comments in `higher_ranked/README.md`.
    ///
    /// The `snapshot` argument to this function is not really used;
    /// it's just there to make it explicit which snapshot bounds the
    /// skolemized region that results. It should always be the top-most snapshot.
    pub fn push_skolemized(&self, br: ty::BoundRegion, snapshot: &RegionSnapshot)
                           -> &'tcx Region {
        assert!(self.in_snapshot());
        assert!(self.undo_log.borrow()[snapshot.length] == OpenSnapshot);

        let sc = self.skolemization_count.get();
        self.skolemization_count.set(sc + 1);
        self.tcx.mk_region(ReSkolemized(ty::SkolemizedRegionVid { index: sc }, br))
    }

    /// Removes all the edges to/from the skolemized regions that are
    /// in `skols`. This is used after a higher-ranked operation
    /// completes to remove all trace of the skolemized regions
    /// created in that time.
    pub fn pop_skolemized(&self,
                          skols: &FxHashSet<&'tcx ty::Region>,
                          snapshot: &RegionSnapshot) {
        debug!("pop_skolemized_regions(skols={:?})", skols);

        assert!(self.in_snapshot());
        assert!(self.undo_log.borrow()[snapshot.length] == OpenSnapshot);
        assert!(self.skolemization_count.get() as usize >= skols.len(),
                "popping more skolemized variables than actually exist, \
                 sc now = {}, skols.len = {}",
                self.skolemization_count.get(),
                skols.len());

        let last_to_pop = self.skolemization_count.get();
        let first_to_pop = last_to_pop - (skols.len() as u32);

        assert!(first_to_pop >= snapshot.skolemization_count,
                "popping more regions than snapshot contains, \
                 sc now = {}, sc then = {}, skols.len = {}",
                self.skolemization_count.get(),
                snapshot.skolemization_count,
                skols.len());
        debug_assert! {
            skols.iter()
                 .all(|&k| match *k {
                     ty::ReSkolemized(index, _) =>
                         index.index >= first_to_pop &&
                         index.index < last_to_pop,
                     _ =>
                         false
                 }),
            "invalid skolemization keys or keys out of range ({}..{}): {:?}",
            snapshot.skolemization_count,
            self.skolemization_count.get(),
            skols
        }

        let mut undo_log = self.undo_log.borrow_mut();

        let constraints_to_kill: Vec<usize> =
            undo_log.iter()
                    .enumerate()
                    .rev()
                    .filter(|&(_, undo_entry)| kill_constraint(skols, undo_entry))
                    .map(|(index, _)| index)
                    .collect();

        for index in constraints_to_kill {
            let undo_entry = mem::replace(&mut undo_log[index], Purged);
            self.rollback_undo_entry(undo_entry);
        }

        self.skolemization_count.set(snapshot.skolemization_count);
        return;

        fn kill_constraint<'tcx>(skols: &FxHashSet<&'tcx ty::Region>,
                                 undo_entry: &UndoLogEntry<'tcx>)
                                 -> bool {
            match undo_entry {
                &AddConstraint(ConstrainVarSubVar(..)) =>
                    false,
                &AddConstraint(ConstrainRegSubVar(a, _)) =>
                    skols.contains(&a),
                &AddConstraint(ConstrainVarSubReg(_, b)) =>
                    skols.contains(&b),
                &AddConstraint(ConstrainRegSubReg(a, b)) =>
                    skols.contains(&a) || skols.contains(&b),
                &AddGiven(..) =>
                    false,
                &AddVerify(_) =>
                    false,
                &AddCombination(_, ref two_regions) =>
                    skols.contains(&two_regions.a) ||
                    skols.contains(&two_regions.b),
                &AddVar(..) |
                &OpenSnapshot |
                &Purged |
                &CommitedSnapshot =>
                    false,
            }
        }

    }

    pub fn new_bound(&self, debruijn: ty::DebruijnIndex) -> &'tcx Region {
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
            bug!("rollover in RegionInference new_bound()");
        }

        self.tcx.mk_region(ReLateBound(debruijn, BrFresh(sc)))
    }

    fn values_are_none(&self) -> bool {
        self.values.borrow().is_none()
    }

    fn add_constraint(&self, constraint: Constraint<'tcx>, origin: SubregionOrigin<'tcx>) {
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
        match verify.bound {
            VerifyBound::AllBounds(ref bs) if bs.len() == 0 => { return; }
            _ => { }
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

    pub fn make_eqregion(&self,
                         origin: SubregionOrigin<'tcx>,
                         sub: &'tcx Region,
                         sup: &'tcx Region) {
        if sub != sup {
            // Eventually, it would be nice to add direct support for
            // equating regions.
            self.make_subregion(origin.clone(), sub, sup);
            self.make_subregion(origin, sup, sub);

            if let (ty::ReVar(sub), ty::ReVar(sup)) = (*sub, *sup) {
                self.unification_table.borrow_mut().union(sub, sup);
            }
        }
    }

    pub fn make_subregion(&self,
                          origin: SubregionOrigin<'tcx>,
                          sub: &'tcx Region,
                          sup: &'tcx Region) {
        // cannot add constraints once regions are resolved
        assert!(self.values_are_none());

        debug!("RegionVarBindings: make_subregion({:?}, {:?}) due to {:?}",
               sub,
               sup,
               origin);

        match (sub, sup) {
            (&ReEarlyBound(..), _) |
            (&ReLateBound(..), _) |
            (_, &ReEarlyBound(..)) |
            (_, &ReLateBound(..)) => {
                span_bug!(origin.span(),
                          "cannot relate bound region: {:?} <= {:?}",
                          sub,
                          sup);
            }
            (_, &ReStatic) => {
                // all regions are subregions of static, so we can ignore this
            }
            (&ReVar(sub_id), &ReVar(sup_id)) => {
                self.add_constraint(ConstrainVarSubVar(sub_id, sup_id), origin);
            }
            (_, &ReVar(sup_id)) => {
                self.add_constraint(ConstrainRegSubVar(sub, sup_id), origin);
            }
            (&ReVar(sub_id), _) => {
                self.add_constraint(ConstrainVarSubReg(sub_id, sup), origin);
            }
            _ => {
                self.add_constraint(ConstrainRegSubReg(sub, sup), origin);
            }
        }
    }

    /// See `Verify::VerifyGenericBound`
    pub fn verify_generic_bound(&self,
                                origin: SubregionOrigin<'tcx>,
                                kind: GenericKind<'tcx>,
                                sub: &'tcx Region,
                                bound: VerifyBound<'tcx>) {
        self.add_verify(Verify {
            kind: kind,
            origin: origin,
            region: sub,
            bound: bound
        });
    }

    pub fn lub_regions(&self,
                       origin: SubregionOrigin<'tcx>,
                       a: &'tcx Region,
                       b: &'tcx Region)
                       -> &'tcx Region {
        // cannot add constraints once regions are resolved
        assert!(self.values_are_none());

        debug!("RegionVarBindings: lub_regions({:?}, {:?})", a, b);
        match (a, b) {
            (r @ &ReStatic, _) | (_, r @ &ReStatic) => {
                r // nothing lives longer than static
            }

            _ if a == b => {
                a // LUB(a,a) = a
            }

            _ => {
                self.combine_vars(Lub, a, b, origin.clone(), |this, old_r, new_r| {
                    this.make_subregion(origin.clone(), old_r, new_r)
                })
            }
        }
    }

    pub fn glb_regions(&self,
                       origin: SubregionOrigin<'tcx>,
                       a: &'tcx Region,
                       b: &'tcx Region)
                       -> &'tcx Region {
        // cannot add constraints once regions are resolved
        assert!(self.values_are_none());

        debug!("RegionVarBindings: glb_regions({:?}, {:?})", a, b);
        match (a, b) {
            (&ReStatic, r) | (r, &ReStatic) => {
                r // static lives longer than everything else
            }

            _ if a == b => {
                a // GLB(a,a) = a
            }

            _ => {
                self.combine_vars(Glb, a, b, origin.clone(), |this, old_r, new_r| {
                    this.make_subregion(origin.clone(), new_r, old_r)
                })
            }
        }
    }

    pub fn resolve_var(&self, rid: RegionVid) -> &'tcx ty::Region {
        match *self.values.borrow() {
            None => {
                span_bug!((*self.var_origins.borrow())[rid.index as usize].span(),
                          "attempt to resolve region variable before values have \
                           been computed!")
            }
            Some(ref values) => {
                let r = lookup(self.tcx, values, rid);
                debug!("resolve_var({:?}) = {:?}", rid, r);
                r
            }
        }
    }

    pub fn opportunistic_resolve_var(&self, rid: RegionVid) -> &'tcx ty::Region {
        let vid = self.unification_table.borrow_mut().find_value(rid).min_vid;
        self.tcx.mk_region(ty::ReVar(vid))
    }

    fn combine_map(&self, t: CombineMapType) -> &RefCell<CombineMap<'tcx>> {
        match t {
            Glb => &self.glbs,
            Lub => &self.lubs,
        }
    }

    pub fn combine_vars<F>(&self,
                           t: CombineMapType,
                           a: &'tcx Region,
                           b: &'tcx Region,
                           origin: SubregionOrigin<'tcx>,
                           mut relate: F)
                           -> &'tcx Region
        where F: FnMut(&RegionVarBindings<'a, 'gcx, 'tcx>, &'tcx Region, &'tcx Region)
    {
        let vars = TwoRegions { a: a, b: b };
        if let Some(&c) = self.combine_map(t).borrow().get(&vars) {
            return self.tcx.mk_region(ReVar(c));
        }
        let c = self.new_region_var(MiscVariable(origin.span()));
        self.combine_map(t).borrow_mut().insert(vars, c);
        if self.in_snapshot() {
            self.undo_log.borrow_mut().push(AddCombination(t, vars));
        }
        relate(self, a, self.tcx.mk_region(ReVar(c)));
        relate(self, b, self.tcx.mk_region(ReVar(c)));
        debug!("combine_vars() c={:?}", c);
        self.tcx.mk_region(ReVar(c))
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

    /// Computes all regions that have been related to `r0` since the
    /// mark `mark` was made---`r0` itself will be the first
    /// entry. The `directions` parameter controls what kind of
    /// relations are considered. For example, one can say that only
    /// "incoming" edges to `r0` are desired, in which case one will
    /// get the set of regions `{r|r <= r0}`. This is used when
    /// checking whether skolemized regions are being improperly
    /// related to other regions.
    pub fn tainted(&self,
                   mark: &RegionSnapshot,
                   r0: &'tcx Region,
                   directions: TaintDirections)
                   -> FxHashSet<&'tcx ty::Region> {
        debug!("tainted(mark={:?}, r0={:?}, directions={:?})",
               mark, r0, directions);

        // `result_set` acts as a worklist: we explore all outgoing
        // edges and add any new regions we find to result_set.  This
        // is not a terribly efficient implementation.
        let mut taint_set = TaintSet::new(directions, r0);
        taint_set.fixed_point(self.tcx,
                              &self.undo_log.borrow()[mark.length..],
                              &self.verifys.borrow());
        debug!("tainted: result={:?}", taint_set.regions);
        return taint_set.into_set();
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

    fn lub_concrete_regions(&self,
                            free_regions: &FreeRegionMap,
                            a: &'tcx Region,
                            b: &'tcx Region)
                            -> &'tcx Region {
        match (a, b) {
            (&ReLateBound(..), _) |
            (_, &ReLateBound(..)) |
            (&ReEarlyBound(..), _) |
            (_, &ReEarlyBound(..)) |
            (&ReErased, _) |
            (_, &ReErased) => {
                bug!("cannot relate region: LUB({:?}, {:?})", a, b);
            }

            (r @ &ReStatic, _) | (_, r @ &ReStatic) => {
                r // nothing lives longer than static
            }

            (&ReEmpty, r) | (r, &ReEmpty) => {
                r // everything lives longer than empty
            }

            (&ReVar(v_id), _) | (_, &ReVar(v_id)) => {
                span_bug!((*self.var_origins.borrow())[v_id.index as usize].span(),
                          "lub_concrete_regions invoked with non-concrete \
                           regions: {:?}, {:?}",
                          a,
                          b);
            }

            (&ReFree(fr), &ReScope(s_id)) |
            (&ReScope(s_id), &ReFree(fr)) => {
                // A "free" region can be interpreted as "some region
                // at least as big as the block fr.scope_id".  So, we can
                // reasonably compare free regions and scopes:
                let r_id = self.tcx.region_maps.nearest_common_ancestor(fr.scope, s_id);

                if r_id == fr.scope {
                    // if the free region's scope `fr.scope_id` is bigger than
                    // the scope region `s_id`, then the LUB is the free
                    // region itself:
                    self.tcx.mk_region(ReFree(fr))
                } else {
                    // otherwise, we don't know what the free region is,
                    // so we must conservatively say the LUB is static:
                    self.tcx.mk_region(ReStatic)
                }
            }

            (&ReScope(a_id), &ReScope(b_id)) => {
                // The region corresponding to an outer block is a
                // subtype of the region corresponding to an inner
                // block.
                self.tcx.mk_region(ReScope(
                    self.tcx.region_maps.nearest_common_ancestor(a_id, b_id)))
            }

            (&ReFree(a_fr), &ReFree(b_fr)) => {
                self.tcx.mk_region(free_regions.lub_free_regions(a_fr, b_fr))
            }

            // For these types, we cannot define any additional
            // relationship:
            (&ReSkolemized(..), _) |
            (_, &ReSkolemized(..)) => {
                if a == b {
                    a
                } else {
                    self.tcx.mk_region(ReStatic)
                }
            }
        }
    }
}

// ______________________________________________________________________

#[derive(Copy, Clone, Debug)]
pub enum VarValue<'tcx> {
    Value(&'tcx Region),
    ErrorValue,
}

struct RegionAndOrigin<'tcx> {
    region: &'tcx Region,
    origin: SubregionOrigin<'tcx>,
}

type RegionGraph<'tcx> = graph::Graph<(), Constraint<'tcx>>;

impl<'a, 'gcx, 'tcx> RegionVarBindings<'a, 'gcx, 'tcx> {
    fn infer_variable_values(&self,
                             free_regions: &FreeRegionMap,
                             errors: &mut Vec<RegionResolutionError<'tcx>>,
                             subject: ast::NodeId)
                             -> Vec<VarValue<'tcx>> {
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
        self.collect_errors(free_regions, &mut var_data, errors);
        self.collect_var_errors(free_regions, &var_data, &graph, errors);
        var_data
    }

    fn construct_var_data(&self) -> Vec<VarValue<'tcx>> {
        (0..self.num_vars() as usize)
            .map(|_| Value(self.tcx.mk_region(ty::ReEmpty)))
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
            for succ_index in graph.depth_traverse(seed_index, OUTGOING) {
                let succ_index = succ_index.0 as u32;
                if succ_index < self.num_vars() {
                    let succ_vid = RegionVid { index: succ_index };
                    givens.insert((fr, succ_vid));
                }
            }
        }
    }

    fn expansion(&self, free_regions: &FreeRegionMap, var_values: &mut [VarValue<'tcx>]) {
        self.iterate_until_fixed_point("Expansion", |constraint, origin| {
            debug!("expansion: constraint={:?} origin={:?}",
                   constraint, origin);
            match *constraint {
                ConstrainRegSubVar(a_region, b_vid) => {
                    let b_data = &mut var_values[b_vid.index as usize];
                    self.expand_node(free_regions, a_region, b_vid, b_data)
                }
                ConstrainVarSubVar(a_vid, b_vid) => {
                    match var_values[a_vid.index as usize] {
                        ErrorValue => false,
                        Value(a_region) => {
                            let b_node = &mut var_values[b_vid.index as usize];
                            self.expand_node(free_regions, a_region, b_vid, b_node)
                        }
                    }
                }
                ConstrainRegSubReg(..) |
                ConstrainVarSubReg(..) => {
                    // These constraints are checked after expansion
                    // is done, in `collect_errors`.
                    false
                }
            }
        })
    }

    fn expand_node(&self,
                   free_regions: &FreeRegionMap,
                   a_region: &'tcx Region,
                   b_vid: RegionVid,
                   b_data: &mut VarValue<'tcx>)
                   -> bool {
        debug!("expand_node({:?}, {:?} == {:?})",
               a_region,
               b_vid,
               b_data);

        // Check if this relationship is implied by a given.
        match *a_region {
            ty::ReFree(fr) => {
                if self.givens.borrow().contains(&(fr, b_vid)) {
                    debug!("given");
                    return false;
                }
            }
            _ => {}
        }

        match *b_data {
            Value(cur_region) => {
                let lub = self.lub_concrete_regions(free_regions, a_region, cur_region);
                if lub == cur_region {
                    return false;
                }

                debug!("Expanding value of {:?} from {:?} to {:?}",
                       b_vid,
                       cur_region,
                       lub);

                *b_data = Value(lub);
                return true;
            }

            ErrorValue => {
                return false;
            }
        }
    }

    /// After expansion is complete, go and check upper bounds (i.e.,
    /// cases where the region cannot grow larger than a fixed point)
    /// and check that they are satisfied.
    fn collect_errors(&self,
                      free_regions: &FreeRegionMap,
                      var_data: &mut Vec<VarValue<'tcx>>,
                      errors: &mut Vec<RegionResolutionError<'tcx>>) {
        let constraints = self.constraints.borrow();
        for (constraint, origin) in constraints.iter() {
            debug!("collect_errors: constraint={:?} origin={:?}",
                   constraint, origin);
            match *constraint {
                ConstrainRegSubVar(..) |
                ConstrainVarSubVar(..) => {
                    // Expansion will ensure that these constraints hold. Ignore.
                }

                ConstrainRegSubReg(sub, sup) => {
                    if free_regions.is_subregion_of(self.tcx, sub, sup) {
                        continue;
                    }

                    debug!("collect_errors: region error at {:?}: \
                            cannot verify that {:?} <= {:?}",
                           origin,
                           sub,
                           sup);

                    errors.push(ConcreteFailure((*origin).clone(), sub, sup));
                }

                ConstrainVarSubReg(a_vid, b_region) => {
                    let a_data = &mut var_data[a_vid.index as usize];
                    debug!("contraction: {:?} == {:?}, {:?}",
                           a_vid,
                           a_data,
                           b_region);

                    let a_region = match *a_data {
                        ErrorValue => continue,
                        Value(a_region) => a_region,
                    };

                    // Do not report these errors immediately:
                    // instead, set the variable value to error and
                    // collect them later.
                    if !free_regions.is_subregion_of(self.tcx, a_region, b_region) {
                        debug!("collect_errors: region error at {:?}: \
                                cannot verify that {:?}={:?} <= {:?}",
                               origin,
                               a_vid,
                               a_region,
                               b_region);
                        *a_data = ErrorValue;
                    }
                }
            }
        }

        for verify in self.verifys.borrow().iter() {
            debug!("collect_errors: verify={:?}", verify);
            let sub = normalize(self.tcx, var_data, verify.region);
            if verify.bound.is_met(self.tcx, free_regions, var_data, sub) {
                continue;
            }

            debug!("collect_errors: region error at {:?}: \
                    cannot verify that {:?} <= {:?}",
                   verify.origin,
                   verify.region,
                   verify.bound);

            errors.push(GenericBoundFailure(verify.origin.clone(),
                                            verify.kind.clone(),
                                            sub));
        }
    }

    /// Go over the variables that were declared to be error variables
    /// and create a `RegionResolutionError` for each of them.
    fn collect_var_errors(&self,
                          free_regions: &FreeRegionMap,
                          var_data: &[VarValue<'tcx>],
                          graph: &RegionGraph<'tcx>,
                          errors: &mut Vec<RegionResolutionError<'tcx>>) {
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
                ConstrainVarSubVar(a_id, b_id) => {
                    graph.add_edge(NodeIndex(a_id.index as usize),
                                   NodeIndex(b_id.index as usize),
                                   *constraint);
                }
                ConstrainRegSubVar(_, b_id) => {
                    graph.add_edge(dummy_source, NodeIndex(b_id.index as usize), *constraint);
                }
                ConstrainVarSubReg(a_id, _) => {
                    graph.add_edge(NodeIndex(a_id.index as usize), dummy_sink, *constraint);
                }
                ConstrainRegSubReg(..) => {
                    // this would be an edge from `dummy_source` to
                    // `dummy_sink`; just ignore it.
                }
            }
        }

        return graph;
    }

    fn collect_error_for_expanding_node(&self,
                                        free_regions: &FreeRegionMap,
                                        graph: &RegionGraph<'tcx>,
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
                (&ReFree(..), &ReFree(..)) => Equal,
                (&ReFree(..), _) => Less,
                (_, &ReFree(..)) => Greater,
                (..) => Equal,
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

        span_bug!((*self.var_origins.borrow())[node_idx.index as usize].span(),
                  "collect_error_for_expanding_node() could not find \
                   error for var {:?}, lower_bounds={:?}, \
                   upper_bounds={:?}",
                  node_idx,
                  lower_bounds,
                  upper_bounds);
    }

    fn collect_concrete_regions(&self,
                                graph: &RegionGraph<'tcx>,
                                orig_node_idx: RegionVid,
                                dir: Direction,
                                dup_vec: &mut [u32])
                                -> (Vec<RegionAndOrigin<'tcx>>, bool) {
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

            debug!("collect_concrete_regions(orig_node_idx={:?}, node_idx={:?})",
                   orig_node_idx,
                   node_idx);

            process_edges(self, &mut state, graph, node_idx, dir);
        }

        let WalkState {result, dup_found, ..} = state;
        return (result, dup_found);

        fn process_edges<'a, 'gcx, 'tcx>(this: &RegionVarBindings<'a, 'gcx, 'tcx>,
                                         state: &mut WalkState<'tcx>,
                                         graph: &RegionGraph<'tcx>,
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

                    ConstrainRegSubReg(..) => {
                        panic!("cannot reach reg-sub-reg edge in region inference \
                                post-processing")
                    }
                }
            }
        }
    }

    fn iterate_until_fixed_point<F>(&self, tag: &str, mut body: F)
        where F: FnMut(&Constraint<'tcx>, &SubregionOrigin<'tcx>) -> bool
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

fn normalize<'a, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                             values: &Vec<VarValue<'tcx>>,
                             r: &'tcx ty::Region)
                             -> &'tcx ty::Region {
    match *r {
        ty::ReVar(rid) => lookup(tcx, values, rid),
        _ => r,
    }
}

fn lookup<'a, 'gcx, 'tcx>(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                          values: &Vec<VarValue<'tcx>>,
                          rid: ty::RegionVid)
                          -> &'tcx ty::Region {
    match values[rid.index as usize] {
        Value(r) => r,
        ErrorValue => tcx.mk_region(ReStatic), // Previously reported error.
    }
}

impl<'tcx> fmt::Debug for RegionAndOrigin<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RegionAndOrigin({:?},{:?})", self.region, self.origin)
    }
}

impl fmt::Debug for RegionSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RegionSnapshot(length={},skolemization={})",
               self.length, self.skolemization_count)
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

impl<'a, 'gcx, 'tcx> GenericKind<'tcx> {
    pub fn to_ty(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx> {
        match *self {
            GenericKind::Param(ref p) => p.to_ty(tcx),
            GenericKind::Projection(ref p) => tcx.mk_projection(p.trait_ref.clone(), p.item_name),
        }
    }
}

impl<'a, 'gcx, 'tcx> VerifyBound<'tcx> {
    fn for_each_region(&self, f: &mut FnMut(&'tcx ty::Region)) {
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
            &VerifyBound::AnyRegion(ref bs) => bs.contains(&&ty::ReStatic),
            &VerifyBound::AllRegions(ref bs) => bs.is_empty(),
            &VerifyBound::AnyBound(ref bs) => bs.iter().any(|b| b.must_hold()),
            &VerifyBound::AllBounds(ref bs) => bs.iter().all(|b| b.must_hold()),
        }
    }

    pub fn cannot_hold(&self) -> bool {
        match self {
            &VerifyBound::AnyRegion(ref bs) => bs.is_empty(),
            &VerifyBound::AllRegions(ref bs) => bs.contains(&&ty::ReEmpty),
            &VerifyBound::AnyBound(ref bs) => bs.iter().all(|b| b.cannot_hold()),
            &VerifyBound::AllBounds(ref bs) => bs.iter().any(|b| b.cannot_hold()),
        }
    }

    pub fn or(self, vb: VerifyBound<'tcx>) -> VerifyBound<'tcx> {
        if self.must_hold() || vb.cannot_hold() {
            self
        } else if self.cannot_hold() || vb.must_hold() {
            vb
        } else {
            VerifyBound::AnyBound(vec![self, vb])
        }
    }

    pub fn and(self, vb: VerifyBound<'tcx>) -> VerifyBound<'tcx> {
        if self.must_hold() && vb.must_hold() {
            self
        } else if self.cannot_hold() && vb.cannot_hold() {
            self
        } else {
            VerifyBound::AllBounds(vec![self, vb])
        }
    }

    fn is_met(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
              free_regions: &FreeRegionMap,
              var_values: &Vec<VarValue<'tcx>>,
              min: &'tcx ty::Region)
              -> bool {
        match self {
            &VerifyBound::AnyRegion(ref rs) =>
                rs.iter()
                  .map(|&r| normalize(tcx, var_values, r))
                  .any(|r| free_regions.is_subregion_of(tcx, min, r)),

            &VerifyBound::AllRegions(ref rs) =>
                rs.iter()
                  .map(|&r| normalize(tcx, var_values, r))
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
