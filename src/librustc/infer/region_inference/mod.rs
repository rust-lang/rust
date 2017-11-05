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
use rustc_data_structures::unify::{self, UnificationTable};
use ty::{self, Ty, TyCtxt};
use ty::{Region, RegionVid};
use ty::ReStatic;
use ty::{ReLateBound, ReVar, ReSkolemized, BrFresh};

use std::collections::BTreeMap;
use std::cell::{Cell, RefCell};
use std::fmt;
use std::mem;
use std::u32;

mod lexical_resolve;
mod graphviz;

/// A constraint that influences the inference process.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Constraint<'tcx> {
    /// One region variable is subregion of another
    ConstrainVarSubVar(RegionVid, RegionVid),

    /// Concrete region is subregion of region variable
    ConstrainRegSubVar(Region<'tcx>, RegionVid),

    /// Region variable is subregion of concrete region. This does not
    /// directly affect inference, but instead is checked after
    /// inference is complete.
    ConstrainVarSubReg(RegionVid, Region<'tcx>),

    /// A constraint where neither side is a variable. This does not
    /// directly affect inference, but instead is checked after
    /// inference is complete.
    ConstrainRegSubReg(Region<'tcx>, Region<'tcx>),
}

/// VerifyGenericBound(T, _, R, RS): The parameter type `T` (or
/// associated type) must outlive the region `R`. `T` is known to
/// outlive `RS`. Therefore verify that `R <= RS[i]` for some
/// `i`. Inference variables may be involved (but this verification
/// step doesn't influence inference).
#[derive(Debug)]
pub struct Verify<'tcx> {
    pub kind: GenericKind<'tcx>,
    pub origin: SubregionOrigin<'tcx>,
    pub region: Region<'tcx>,
    pub bound: VerifyBound<'tcx>,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum GenericKind<'tcx> {
    Param(ty::ParamTy),
    Projection(ty::ProjectionTy<'tcx>),
}

/// When we introduce a verification step, we wish to test that a
/// particular region (let's call it `'min`) meets some bound.
/// The bound is described the by the following grammar:
#[derive(Debug)]
pub enum VerifyBound<'tcx> {
    /// B = exists {R} --> some 'r in {R} must outlive 'min
    ///
    /// Put another way, the subject value is known to outlive all
    /// regions in {R}, so if any of those outlives 'min, then the
    /// bound is met.
    AnyRegion(Vec<Region<'tcx>>),

    /// B = forall {R} --> all 'r in {R} must outlive 'min
    ///
    /// Put another way, the subject value is known to outlive some
    /// region in {R}, so if all of those outlives 'min, then the bound
    /// is met.
    AllRegions(Vec<Region<'tcx>>),

    /// B = exists {B} --> 'min must meet some bound b in {B}
    AnyBound(Vec<VerifyBound<'tcx>>),

    /// B = forall {B} --> 'min must meet all bounds b in {B}
    AllBounds(Vec<VerifyBound<'tcx>>),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct TwoRegions<'tcx> {
    a: Region<'tcx>,
    b: Region<'tcx>,
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
    AddGiven(Region<'tcx>, ty::RegionVid),

    /// We added a GLB/LUB "combination variable"
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
    SubSupConflict(RegionVariableOrigin,
                   SubregionOrigin<'tcx>,
                   Region<'tcx>,
                   SubregionOrigin<'tcx>,
                   Region<'tcx>),
}

#[derive(Clone, Debug)]
pub enum ProcessedErrorOrigin<'tcx> {
    ConcreteFailure(SubregionOrigin<'tcx>, Region<'tcx>, Region<'tcx>),
    VariableFailure(RegionVariableOrigin),
}

#[derive(Copy, Clone, Debug)]
pub enum VarValue<'tcx> {
    Value(Region<'tcx>),
    ErrorValue,
}

pub type CombineMap<'tcx> = FxHashMap<TwoRegions<'tcx>, RegionVid>;

pub struct RegionVarBindings<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    var_origins: RefCell<Vec<RegionVariableOrigin>>,

    /// Constraints of the form `A <= B` introduced by the region
    /// checker.  Here at least one of `A` and `B` must be a region
    /// variable.
    ///
    /// Using `BTreeMap` because the order in which we iterate over
    /// these constraints can affect the way we build the region graph,
    /// which in turn affects the way that region errors are reported,
    /// leading to small variations in error output across runs and
    /// platforms.
    constraints: RefCell<BTreeMap<Constraint<'tcx>, SubregionOrigin<'tcx>>>,

    /// A "verify" is something that we need to verify after inference is
    /// done, but which does not directly affect inference in any way.
    ///
    /// An example is a `A <= B` where neither `A` nor `B` are
    /// inference variables.
    verifys: RefCell<Vec<Verify<'tcx>>>,

    /// A "given" is a relationship that is known to hold. In particular,
    /// we often know from closure fn signatures that a particular free
    /// region must be a subregion of a region variable:
    ///
    ///    foo.iter().filter(<'a> |x: &'a &'b T| ...)
    ///
    /// In situations like this, `'b` is in fact a region variable
    /// introduced by the call to `iter()`, and `'a` is a bound region
    /// on the closure (as indicated by the `<'a>` prefix). If we are
    /// naive, we wind up inferring that `'b` must be `'static`,
    /// because we require that it be greater than `'a` and we do not
    /// know what `'a` is precisely.
    ///
    /// This hashmap is used to avoid that naive scenario. Basically we
    /// record the fact that `'a <= 'b` is implied by the fn signature,
    /// and then ignore the constraint when solving equations. This is
    /// a bit of a hack but seems to work.
    givens: RefCell<FxHashSet<(Region<'tcx>, ty::RegionVid)>>,

    lubs: RefCell<CombineMap<'tcx>>,
    glbs: RefCell<CombineMap<'tcx>>,
    skolemization_count: Cell<u32>,
    bound_count: Cell<u32>,

    /// The undo log records actions that might later be undone.
    ///
    /// Note: when the undo_log is empty, we are not actively
    /// snapshotting. When the `start_snapshot()` method is called, we
    /// push an OpenSnapshot entry onto the list to indicate that we
    /// are now actively snapshotting. The reason for this is that
    /// otherwise we end up adding entries for things like the lower
    /// bound on a variable and so forth, which can never be rolled
    /// back.
    undo_log: RefCell<Vec<UndoLogEntry<'tcx>>>,

    unification_table: RefCell<UnificationTable<ty::RegionVid>>,

    /// This contains the results of inference.  It begins as an empty
    /// option and only acquires a value after inference is complete.
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
    regions: FxHashSet<ty::Region<'tcx>>
}

impl<'a, 'gcx, 'tcx> TaintSet<'tcx> {
    fn new(directions: TaintDirections,
           initial_region: ty::Region<'tcx>)
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
                        self.add_edge(a, tcx.mk_region(ReVar(b)));
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

    fn into_set(self) -> FxHashSet<ty::Region<'tcx>> {
        self.regions
    }

    fn len(&self) -> usize {
        self.regions.len()
    }

    fn add_edge(&mut self,
                source: ty::Region<'tcx>,
                target: ty::Region<'tcx>) {
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
            tcx,
            var_origins: RefCell::new(Vec::new()),
            values: RefCell::new(None),
            constraints: RefCell::new(BTreeMap::new()),
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
            length,
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
                           -> Region<'tcx> {
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
                          skols: &FxHashSet<ty::Region<'tcx>>,
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

        fn kill_constraint<'tcx>(skols: &FxHashSet<ty::Region<'tcx>>,
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

    pub fn new_bound(&self, debruijn: ty::DebruijnIndex) -> Region<'tcx> {
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

        // never overwrite an existing (constraint, origin) - only insert one if it isn't
        // present in the map yet. This prevents origins from outside the snapshot being
        // replaced with "less informative" origins e.g. during calls to `can_eq`
        self.constraints.borrow_mut().entry(constraint).or_insert_with(|| {
            if self.in_snapshot() {
                self.undo_log.borrow_mut().push(AddConstraint(constraint));
            }
            origin
        });
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

    pub fn add_given(&self, sub: Region<'tcx>, sup: ty::RegionVid) {
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
                         sub: Region<'tcx>,
                         sup: Region<'tcx>) {
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
                          sub: Region<'tcx>,
                          sup: Region<'tcx>) {
        // cannot add constraints once regions are resolved
        assert!(self.values_are_none());

        debug!("RegionVarBindings: make_subregion({:?}, {:?}) due to {:?}",
               sub,
               sup,
               origin);

        match (sub, sup) {
            (&ReLateBound(..), _) |
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
                                sub: Region<'tcx>,
                                bound: VerifyBound<'tcx>) {
        self.add_verify(Verify {
            kind,
            origin,
            region: sub,
            bound,
        });
    }

    pub fn lub_regions(&self,
                       origin: SubregionOrigin<'tcx>,
                       a: Region<'tcx>,
                       b: Region<'tcx>)
                       -> Region<'tcx> {
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
                       a: Region<'tcx>,
                       b: Region<'tcx>)
                       -> Region<'tcx> {
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

    pub fn opportunistic_resolve_var(&self, rid: RegionVid) -> ty::Region<'tcx> {
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
                           a: Region<'tcx>,
                           b: Region<'tcx>,
                           origin: SubregionOrigin<'tcx>,
                           mut relate: F)
                           -> Region<'tcx>
        where F: FnMut(&RegionVarBindings<'a, 'gcx, 'tcx>, Region<'tcx>, Region<'tcx>)
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
                   r0: Region<'tcx>,
                   directions: TaintDirections)
                   -> FxHashSet<ty::Region<'tcx>> {
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
            GenericKind::Projection(ref p) => tcx.mk_projection(p.item_def_id, p.substs),
        }
    }
}

impl<'a, 'gcx, 'tcx> VerifyBound<'tcx> {
    fn for_each_region(&self, f: &mut FnMut(ty::Region<'tcx>)) {
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
}
