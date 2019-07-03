//! See `README.md`.

use self::CombineMapType::*;
use self::UndoLog::*;

use super::unify_key;
use super::{MiscVariable, RegionVariableOrigin, SubregionOrigin};

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::sync::Lrc;
use rustc_data_structures::unify as ut;
use crate::hir::def_id::DefId;
use crate::ty::ReStatic;
use crate::ty::{self, Ty, TyCtxt};
use crate::ty::{ReLateBound, ReVar};
use crate::ty::{Region, RegionVid};
use syntax_pos::Span;

use std::collections::BTreeMap;
use std::{cmp, fmt, mem};
use std::ops::Range;

mod leak_check;

#[derive(Default)]
pub struct RegionConstraintCollector<'tcx> {
    /// For each `RegionVid`, the corresponding `RegionVariableOrigin`.
    var_infos: IndexVec<RegionVid, RegionVariableInfo>,

    data: RegionConstraintData<'tcx>,

    /// For a given pair of regions (R1, R2), maps to a region R3 that
    /// is designated as their LUB (edges R1 <= R3 and R2 <= R3
    /// exist). This prevents us from making many such regions.
    lubs: CombineMap<'tcx>,

    /// For a given pair of regions (R1, R2), maps to a region R3 that
    /// is designated as their GLB (edges R3 <= R1 and R3 <= R2
    /// exist). This prevents us from making many such regions.
    glbs: CombineMap<'tcx>,

    /// The undo log records actions that might later be undone.
    ///
    /// Note: `num_open_snapshots` is used to track if we are actively
    /// snapshotting. When the `start_snapshot()` method is called, we
    /// increment `num_open_snapshots` to indicate that we are now actively
    /// snapshotting. The reason for this is that otherwise we end up adding
    /// entries for things like the lower bound on a variable and so forth,
    /// which can never be rolled back.
    undo_log: Vec<UndoLog<'tcx>>,

    /// The number of open snapshots, i.e., those that haven't been committed or
    /// rolled back.
    num_open_snapshots: usize,

    /// When we add a R1 == R2 constriant, we currently add (a) edges
    /// R1 <= R2 and R2 <= R1 and (b) we unify the two regions in this
    /// table. You can then call `opportunistic_resolve_var` early
    /// which will map R1 and R2 to some common region (i.e., either
    /// R1 or R2). This is important when dropck and other such code
    /// is iterating to a fixed point, because otherwise we sometimes
    /// would wind up with a fresh stream of region variables that
    /// have been equated but appear distinct.
    unification_table: ut::UnificationTable<ut::InPlace<ty::RegionVid>>,

    /// a flag set to true when we perform any unifications; this is used
    /// to micro-optimize `take_and_reset_data`
    any_unifications: bool,
}

pub type VarInfos = IndexVec<RegionVid, RegionVariableInfo>;

/// The full set of region constraints gathered up by the collector.
/// Describes constraints between the region variables and other
/// regions, as well as other conditions that must be verified, or
/// assumptions that can be made.
#[derive(Debug, Default, Clone)]
pub struct RegionConstraintData<'tcx> {
    /// Constraints of the form `A <= B`, where either `A` or `B` can
    /// be a region variable (or neither, as it happens).
    pub constraints: BTreeMap<Constraint<'tcx>, SubregionOrigin<'tcx>>,

    /// Constraints of the form `R0 member of [R1, ..., Rn]`, meaning that
    /// `R0` must be equal to one of the regions `R1..Rn`. These occur
    /// with `impl Trait` quite frequently.
    pub member_constraints: Vec<MemberConstraint<'tcx>>,

    /// A "verify" is something that we need to verify after inference
    /// is done, but which does not directly affect inference in any
    /// way.
    ///
    /// An example is a `A <= B` where neither `A` nor `B` are
    /// inference variables.
    pub verifys: Vec<Verify<'tcx>>,

    /// A "given" is a relationship that is known to hold. In
    /// particular, we often know from closure fn signatures that a
    /// particular free region must be a subregion of a region
    /// variable:
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
    /// This hashmap is used to avoid that naive scenario. Basically
    /// we record the fact that `'a <= 'b` is implied by the fn
    /// signature, and then ignore the constraint when solving
    /// equations. This is a bit of a hack but seems to work.
    pub givens: FxHashSet<(Region<'tcx>, ty::RegionVid)>,
}

/// Represents a constraint that influences the inference process.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Constraint<'tcx> {
    /// A region variable is a subregion of another.
    VarSubVar(RegionVid, RegionVid),

    /// A concrete region is a subregion of region variable.
    RegSubVar(Region<'tcx>, RegionVid),

    /// A region variable is a subregion of a concrete region. This does not
    /// directly affect inference, but instead is checked after
    /// inference is complete.
    VarSubReg(RegionVid, Region<'tcx>),

    /// A constraint where neither side is a variable. This does not
    /// directly affect inference, but instead is checked after
    /// inference is complete.
    RegSubReg(Region<'tcx>, Region<'tcx>),
}

impl Constraint<'_> {
    pub fn involves_placeholders(&self) -> bool {
        match self {
            Constraint::VarSubVar(_, _) => false,
            Constraint::VarSubReg(_, r) | Constraint::RegSubVar(r, _) => r.is_placeholder(),
            Constraint::RegSubReg(r, s) => r.is_placeholder() || s.is_placeholder(),
        }
    }
}

/// Requires that `region` must be equal to one of the regions in `choice_regions`.
/// We often denote this using the syntax:
///
/// ```
/// R0 member of [O1..On]
/// ```
#[derive(Debug, Clone, HashStable)]
pub struct MemberConstraint<'tcx> {
    /// The `DefId` of the opaque type causing this constraint: used for error reporting.
    pub opaque_type_def_id: DefId,

    /// The span where the hidden type was instantiated.
    pub definition_span: Span,

    /// The hidden type in which `member_region` appears: used for error reporting.
    pub hidden_ty: Ty<'tcx>,

    /// The region `R0`.
    pub member_region: Region<'tcx>,

    /// The options `O1..On`.
    pub choice_regions: Lrc<Vec<Region<'tcx>>>,
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for MemberConstraint<'tcx> {
        opaque_type_def_id, definition_span, hidden_ty, member_region, choice_regions
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for MemberConstraint<'a> {
        type Lifted = MemberConstraint<'tcx>;
        opaque_type_def_id, definition_span, hidden_ty, member_region, choice_regions
    }
}

/// `VerifyGenericBound(T, _, R, RS)`: the parameter type `T` (or
/// associated type) must outlive the region `R`. `T` is known to
/// outlive `RS`. Therefore, verify that `R <= RS[i]` for some
/// `i`. Inference variables may be involved (but this verification
/// step doesn't influence inference).
#[derive(Debug, Clone)]
pub struct Verify<'tcx> {
    pub kind: GenericKind<'tcx>,
    pub origin: SubregionOrigin<'tcx>,
    pub region: Region<'tcx>,
    pub bound: VerifyBound<'tcx>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum GenericKind<'tcx> {
    Param(ty::ParamTy),
    Projection(ty::ProjectionTy<'tcx>),
}

EnumTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for GenericKind<'tcx> {
        (GenericKind::Param)(a),
        (GenericKind::Projection)(a),
    }
}

/// Describes the things that some `GenericKind` value `G` is known to
/// outlive. Each variant of `VerifyBound` can be thought of as a
/// function:
///
///     fn(min: Region) -> bool { .. }
///
/// where `true` means that the region `min` meets that `G: min`.
/// (False means nothing.)
///
/// So, for example, if we have the type `T` and we have in scope that
/// `T: 'a` and `T: 'b`, then the verify bound might be:
///
///     fn(min: Region) -> bool {
///        ('a: min) || ('b: min)
///     }
///
/// This is described with a `AnyRegion('a, 'b)` node.
#[derive(Debug, Clone)]
pub enum VerifyBound<'tcx> {
    /// Given a kind K and a bound B, expands to a function like the
    /// following, where `G` is the generic for which this verify
    /// bound was created:
    ///
    /// ```rust
    /// fn(min) -> bool {
    ///     if G == K {
    ///         B(min)
    ///     } else {
    ///         false
    ///     }
    /// }
    /// ```
    ///
    /// In other words, if the generic `G` that we are checking is
    /// equal to `K`, then check the associated verify bound
    /// (otherwise, false).
    ///
    /// This is used when we have something in the environment that
    /// may or may not be relevant, depending on the region inference
    /// results. For example, we may have `where <T as
    /// Trait<'a>>::Item: 'b` in our where-clauses. If we are
    /// generating the verify-bound for `<T as Trait<'0>>::Item`, then
    /// this where-clause is only relevant if `'0` winds up inferred
    /// to `'a`.
    ///
    /// So we would compile to a verify-bound like
    ///
    /// ```
    /// IfEq(<T as Trait<'a>>::Item, AnyRegion('a))
    /// ```
    ///
    /// meaning, if the subject G is equal to `<T as Trait<'a>>::Item`
    /// (after inference), and `'a: min`, then `G: min`.
    IfEq(Ty<'tcx>, Box<VerifyBound<'tcx>>),

    /// Given a region `R`, expands to the function:
    ///
    /// ```
    /// fn(min) -> bool {
    ///     R: min
    /// }
    /// ```
    ///
    /// This is used when we can establish that `G: R` -- therefore,
    /// if `R: min`, then by transitivity `G: min`.
    OutlivedBy(Region<'tcx>),

    /// Given a set of bounds `B`, expands to the function:
    ///
    /// ```rust
    /// fn(min) -> bool {
    ///     exists (b in B) { b(min) }
    /// }
    /// ```
    ///
    /// In other words, if we meet some bound in `B`, that suffices.
    /// This is used when all the bounds in `B` are known to apply to `G`.
    AnyBound(Vec<VerifyBound<'tcx>>),

    /// Given a set of bounds `B`, expands to the function:
    ///
    /// ```rust
    /// fn(min) -> bool {
    ///     forall (b in B) { b(min) }
    /// }
    /// ```
    ///
    /// In other words, if we meet *all* bounds in `B`, that suffices.
    /// This is used when *some* bound in `B` is known to suffice, but
    /// we don't know which.
    AllBounds(Vec<VerifyBound<'tcx>>),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct TwoRegions<'tcx> {
    a: Region<'tcx>,
    b: Region<'tcx>,
}

#[derive(Copy, Clone, PartialEq)]
enum UndoLog<'tcx> {
    /// We added `RegionVid`.
    AddVar(RegionVid),

    /// We added the given `constraint`.
    AddConstraint(Constraint<'tcx>),

    /// We added the given `verify`.
    AddVerify(usize),

    /// We added the given `given`.
    AddGiven(Region<'tcx>, ty::RegionVid),

    /// We added a GLB/LUB "combination variable".
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
enum CombineMapType {
    Lub,
    Glb,
}

type CombineMap<'tcx> = FxHashMap<TwoRegions<'tcx>, RegionVid>;

#[derive(Debug, Clone, Copy)]
pub struct RegionVariableInfo {
    pub origin: RegionVariableOrigin,
    pub universe: ty::UniverseIndex,
}

pub struct RegionSnapshot {
    length: usize,
    region_snapshot: ut::Snapshot<ut::InPlace<ty::RegionVid>>,
    any_unifications: bool,
}

/// When working with placeholder regions, we often wish to find all of
/// the regions that are either reachable from a placeholder region, or
/// which can reach a placeholder region, or both. We call such regions
/// *tainted* regions. This struct allows you to decide what set of
/// tainted regions you want.
#[derive(Debug)]
pub struct TaintDirections {
    incoming: bool,
    outgoing: bool,
}

impl TaintDirections {
    pub fn incoming() -> Self {
        TaintDirections {
            incoming: true,
            outgoing: false,
        }
    }

    pub fn outgoing() -> Self {
        TaintDirections {
            incoming: false,
            outgoing: true,
        }
    }

    pub fn both() -> Self {
        TaintDirections {
            incoming: true,
            outgoing: true,
        }
    }
}

pub struct ConstraintInfo {}

impl<'tcx> RegionConstraintCollector<'tcx> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn num_region_vars(&self) -> usize {
        self.var_infos.len()
    }

    pub fn region_constraint_data(&self) -> &RegionConstraintData<'tcx> {
        &self.data
    }

    /// Once all the constraints have been gathered, extract out the final data.
    ///
    /// Not legal during a snapshot.
    pub fn into_infos_and_data(self) -> (VarInfos, RegionConstraintData<'tcx>) {
        assert!(!self.in_snapshot());
        (self.var_infos, self.data)
    }

    /// Takes (and clears) the current set of constraints. Note that
    /// the set of variables remains intact, but all relationships
    /// between them are reset. This is used during NLL checking to
    /// grab the set of constraints that arose from a particular
    /// operation.
    ///
    /// We don't want to leak relationships between variables between
    /// points because just because (say) `r1 == r2` was true at some
    /// point P in the graph doesn't imply that it will be true at
    /// some other point Q, in NLL.
    ///
    /// Not legal during a snapshot.
    pub fn take_and_reset_data(&mut self) -> RegionConstraintData<'tcx> {
        assert!(!self.in_snapshot());

        // If you add a new field to `RegionConstraintCollector`, you
        // should think carefully about whether it needs to be cleared
        // or updated in some way.
        let RegionConstraintCollector {
            var_infos: _,
            data,
            lubs,
            glbs,
            undo_log: _,
            num_open_snapshots: _,
            unification_table,
            any_unifications,
        } = self;

        // Clear the tables of (lubs, glbs), so that we will create
        // fresh regions if we do a LUB operation. As it happens,
        // LUB/GLB are not performed by the MIR type-checker, which is
        // the one that uses this method, but it's good to be correct.
        lubs.clear();
        glbs.clear();

        // Clear all unifications and recreate the variables a "now
        // un-unified" state. Note that when we unify `a` and `b`, we
        // also insert `a <= b` and a `b <= a` edges, so the
        // `RegionConstraintData` contains the relationship here.
        if *any_unifications {
            unification_table.reset_unifications(|vid| unify_key::RegionVidKey { min_vid: vid });
            *any_unifications = false;
        }

        mem::take(data)
    }

    pub fn data(&self) -> &RegionConstraintData<'tcx> {
        &self.data
    }

    fn in_snapshot(&self) -> bool {
        self.num_open_snapshots > 0
    }

    pub fn start_snapshot(&mut self) -> RegionSnapshot {
        let length = self.undo_log.len();
        debug!("RegionConstraintCollector: start_snapshot({})", length);
        self.num_open_snapshots += 1;
        RegionSnapshot {
            length,
            region_snapshot: self.unification_table.snapshot(),
            any_unifications: self.any_unifications,
        }
    }

    fn assert_open_snapshot(&self, snapshot: &RegionSnapshot) {
        assert!(self.undo_log.len() >= snapshot.length);
        assert!(self.num_open_snapshots > 0);
    }

    pub fn commit(&mut self, snapshot: RegionSnapshot) {
        debug!("RegionConstraintCollector: commit({})", snapshot.length);
        self.assert_open_snapshot(&snapshot);

        if self.num_open_snapshots == 1 {
            // The root snapshot. It's safe to clear the undo log because
            // there's no snapshot further out that we might need to roll back
            // to.
            assert!(snapshot.length == 0);
            self.undo_log.clear();
        }

        self.num_open_snapshots -= 1;

        self.unification_table.commit(snapshot.region_snapshot);
    }

    pub fn rollback_to(&mut self, snapshot: RegionSnapshot) {
        debug!("RegionConstraintCollector: rollback_to({:?})", snapshot);
        self.assert_open_snapshot(&snapshot);

        while self.undo_log.len() > snapshot.length {
            let undo_entry = self.undo_log.pop().unwrap();
            self.rollback_undo_entry(undo_entry);
        }

        self.num_open_snapshots -= 1;

        self.unification_table.rollback_to(snapshot.region_snapshot);
        self.any_unifications = snapshot.any_unifications;
    }

    fn rollback_undo_entry(&mut self, undo_entry: UndoLog<'tcx>) {
        match undo_entry {
            Purged => {
                // nothing to do here
            }
            AddVar(vid) => {
                self.var_infos.pop().unwrap();
                assert_eq!(self.var_infos.len(), vid.index() as usize);
            }
            AddConstraint(ref constraint) => {
                self.data.constraints.remove(constraint);
            }
            AddVerify(index) => {
                self.data.verifys.pop();
                assert_eq!(self.data.verifys.len(), index);
            }
            AddGiven(sub, sup) => {
                self.data.givens.remove(&(sub, sup));
            }
            AddCombination(Glb, ref regions) => {
                self.glbs.remove(regions);
            }
            AddCombination(Lub, ref regions) => {
                self.lubs.remove(regions);
            }
        }
    }

    pub fn new_region_var(
        &mut self,
        universe: ty::UniverseIndex,
        origin: RegionVariableOrigin,
    ) -> RegionVid {
        let vid = self.var_infos.push(RegionVariableInfo { origin, universe });

        let u_vid = self
            .unification_table
            .new_key(unify_key::RegionVidKey { min_vid: vid });
        assert_eq!(vid, u_vid);
        if self.in_snapshot() {
            self.undo_log.push(AddVar(vid));
        }
        debug!(
            "created new region variable {:?} in {:?} with origin {:?}",
            vid, universe, origin
        );
        return vid;
    }

    /// Returns the universe for the given variable.
    pub fn var_universe(&self, vid: RegionVid) -> ty::UniverseIndex {
        self.var_infos[vid].universe
    }

    /// Returns the origin for the given variable.
    pub fn var_origin(&self, vid: RegionVid) -> RegionVariableOrigin {
        self.var_infos[vid].origin
    }

    /// Removes all the edges to/from the placeholder regions that are
    /// in `skols`. This is used after a higher-ranked operation
    /// completes to remove all trace of the placeholder regions
    /// created in that time.
    pub fn pop_placeholders(&mut self, placeholders: &FxHashSet<ty::Region<'tcx>>) {
        debug!("pop_placeholders(placeholders={:?})", placeholders);

        assert!(self.in_snapshot());

        let constraints_to_kill: Vec<usize> = self
            .undo_log
            .iter()
            .enumerate()
            .rev()
            .filter(|&(_, undo_entry)| kill_constraint(placeholders, undo_entry))
            .map(|(index, _)| index)
            .collect();

        for index in constraints_to_kill {
            let undo_entry = mem::replace(&mut self.undo_log[index], Purged);
            self.rollback_undo_entry(undo_entry);
        }

        return;

        fn kill_constraint<'tcx>(
            placeholders: &FxHashSet<ty::Region<'tcx>>,
            undo_entry: &UndoLog<'tcx>,
        ) -> bool {
            match undo_entry {
                &AddConstraint(Constraint::VarSubVar(..)) => false,
                &AddConstraint(Constraint::RegSubVar(a, _)) => placeholders.contains(&a),
                &AddConstraint(Constraint::VarSubReg(_, b)) => placeholders.contains(&b),
                &AddConstraint(Constraint::RegSubReg(a, b)) => {
                    placeholders.contains(&a) || placeholders.contains(&b)
                }
                &AddGiven(..) => false,
                &AddVerify(_) => false,
                &AddCombination(_, ref two_regions) => {
                    placeholders.contains(&two_regions.a) || placeholders.contains(&two_regions.b)
                }
                &AddVar(..) | &Purged => false,
            }
        }
    }

    fn add_constraint(&mut self, constraint: Constraint<'tcx>, origin: SubregionOrigin<'tcx>) {
        // cannot add constraints once regions are resolved
        debug!(
            "RegionConstraintCollector: add_constraint({:?})",
            constraint
        );

        // never overwrite an existing (constraint, origin) - only insert one if it isn't
        // present in the map yet. This prevents origins from outside the snapshot being
        // replaced with "less informative" origins e.g., during calls to `can_eq`
        let in_snapshot = self.in_snapshot();
        let undo_log = &mut self.undo_log;
        self.data.constraints.entry(constraint).or_insert_with(|| {
            if in_snapshot {
                undo_log.push(AddConstraint(constraint));
            }
            origin
        });
    }

    fn add_verify(&mut self, verify: Verify<'tcx>) {
        // cannot add verifys once regions are resolved
        debug!("RegionConstraintCollector: add_verify({:?})", verify);

        // skip no-op cases known to be satisfied
        if let VerifyBound::AllBounds(ref bs) = verify.bound {
            if bs.len() == 0 {
                return;
            }
        }

        let index = self.data.verifys.len();
        self.data.verifys.push(verify);
        if self.in_snapshot() {
            self.undo_log.push(AddVerify(index));
        }
    }

    pub fn add_given(&mut self, sub: Region<'tcx>, sup: ty::RegionVid) {
        // cannot add givens once regions are resolved
        if self.data.givens.insert((sub, sup)) {
            debug!("add_given({:?} <= {:?})", sub, sup);

            if self.in_snapshot() {
                self.undo_log.push(AddGiven(sub, sup));
            }
        }
    }

    pub fn make_eqregion(
        &mut self,
        origin: SubregionOrigin<'tcx>,
        sub: Region<'tcx>,
        sup: Region<'tcx>,
    ) {
        if sub != sup {
            // Eventually, it would be nice to add direct support for
            // equating regions.
            self.make_subregion(origin.clone(), sub, sup);
            self.make_subregion(origin, sup, sub);

            if let (ty::ReVar(sub), ty::ReVar(sup)) = (*sub, *sup) {
                debug!("make_eqregion: uniying {:?} with {:?}", sub, sup);
                self.unification_table.union(sub, sup);
                self.any_unifications = true;
            }
        }
    }

    pub fn member_constraint(
        &mut self,
        opaque_type_def_id: DefId,
        definition_span: Span,
        hidden_ty: Ty<'tcx>,
        member_region: ty::Region<'tcx>,
        choice_regions: &Lrc<Vec<ty::Region<'tcx>>>,
    ) {
        debug!("member_constraint({:?} in {:#?})", member_region, choice_regions);

        if choice_regions.iter().any(|&r| r == member_region) {
            return;
        }

        self.data.member_constraints.push(MemberConstraint {
            opaque_type_def_id,
            definition_span,
            hidden_ty,
            member_region,
            choice_regions: choice_regions.clone()
        });

    }

    pub fn make_subregion(
        &mut self,
        origin: SubregionOrigin<'tcx>,
        sub: Region<'tcx>,
        sup: Region<'tcx>,
    ) {
        // cannot add constraints once regions are resolved
        debug!(
            "RegionConstraintCollector: make_subregion({:?}, {:?}) due to {:?}",
            sub, sup, origin
        );

        match (sub, sup) {
            (&ReLateBound(..), _) | (_, &ReLateBound(..)) => {
                span_bug!(
                    origin.span(),
                    "cannot relate bound region: {:?} <= {:?}",
                    sub,
                    sup
                );
            }
            (_, &ReStatic) => {
                // all regions are subregions of static, so we can ignore this
            }
            (&ReVar(sub_id), &ReVar(sup_id)) => {
                self.add_constraint(Constraint::VarSubVar(sub_id, sup_id), origin);
            }
            (_, &ReVar(sup_id)) => {
                self.add_constraint(Constraint::RegSubVar(sub, sup_id), origin);
            }
            (&ReVar(sub_id), _) => {
                self.add_constraint(Constraint::VarSubReg(sub_id, sup), origin);
            }
            _ => {
                self.add_constraint(Constraint::RegSubReg(sub, sup), origin);
            }
        }
    }

    /// See [`Verify::VerifyGenericBound`].
    pub fn verify_generic_bound(
        &mut self,
        origin: SubregionOrigin<'tcx>,
        kind: GenericKind<'tcx>,
        sub: Region<'tcx>,
        bound: VerifyBound<'tcx>,
    ) {
        self.add_verify(Verify {
            kind,
            origin,
            region: sub,
            bound,
        });
    }

    pub fn lub_regions(
        &mut self,
        tcx: TyCtxt<'tcx>,
        origin: SubregionOrigin<'tcx>,
        a: Region<'tcx>,
        b: Region<'tcx>,
    ) -> Region<'tcx> {
        // cannot add constraints once regions are resolved
        debug!("RegionConstraintCollector: lub_regions({:?}, {:?})", a, b);
        match (a, b) {
            (r @ &ReStatic, _) | (_, r @ &ReStatic) => {
                r // nothing lives longer than static
            }

            _ if a == b => {
                a // LUB(a,a) = a
            }

            _ => self.combine_vars(tcx, Lub, a, b, origin),
        }
    }

    pub fn glb_regions(
        &mut self,
        tcx: TyCtxt<'tcx>,
        origin: SubregionOrigin<'tcx>,
        a: Region<'tcx>,
        b: Region<'tcx>,
    ) -> Region<'tcx> {
        // cannot add constraints once regions are resolved
        debug!("RegionConstraintCollector: glb_regions({:?}, {:?})", a, b);
        match (a, b) {
            (&ReStatic, r) | (r, &ReStatic) => {
                r // static lives longer than everything else
            }

            _ if a == b => {
                a // GLB(a,a) = a
            }

            _ => self.combine_vars(tcx, Glb, a, b, origin),
        }
    }

    pub fn opportunistic_resolve_var(
        &mut self,
        tcx: TyCtxt<'tcx>,
        rid: RegionVid,
    ) -> ty::Region<'tcx> {
        let vid = self.unification_table.probe_value(rid).min_vid;
        tcx.mk_region(ty::ReVar(vid))
    }

    fn combine_map(&mut self, t: CombineMapType) -> &mut CombineMap<'tcx> {
        match t {
            Glb => &mut self.glbs,
            Lub => &mut self.lubs,
        }
    }

    fn combine_vars(
        &mut self,
        tcx: TyCtxt<'tcx>,
        t: CombineMapType,
        a: Region<'tcx>,
        b: Region<'tcx>,
        origin: SubregionOrigin<'tcx>,
    ) -> Region<'tcx> {
        let vars = TwoRegions { a: a, b: b };
        if let Some(&c) = self.combine_map(t).get(&vars) {
            return tcx.mk_region(ReVar(c));
        }
        let a_universe = self.universe(a);
        let b_universe = self.universe(b);
        let c_universe = cmp::max(a_universe, b_universe);
        let c = self.new_region_var(c_universe, MiscVariable(origin.span()));
        self.combine_map(t).insert(vars, c);
        if self.in_snapshot() {
            self.undo_log.push(AddCombination(t, vars));
        }
        let new_r = tcx.mk_region(ReVar(c));
        for &old_r in &[a, b] {
            match t {
                Glb => self.make_subregion(origin.clone(), new_r, old_r),
                Lub => self.make_subregion(origin.clone(), old_r, new_r),
            }
        }
        debug!("combine_vars() c={:?}", c);
        new_r
    }

    pub fn universe(&self, region: Region<'tcx>) -> ty::UniverseIndex {
        match *region {
            ty::ReScope(..)
            | ty::ReStatic
            | ty::ReEmpty
            | ty::ReErased
            | ty::ReFree(..)
            | ty::ReEarlyBound(..) => ty::UniverseIndex::ROOT,
            ty::RePlaceholder(placeholder) => placeholder.universe,
            ty::ReClosureBound(vid) | ty::ReVar(vid) => self.var_universe(vid),
            ty::ReLateBound(..) => bug!("universe(): encountered bound region {:?}", region),
        }
    }

    pub fn vars_since_snapshot(
        &self,
        mark: &RegionSnapshot,
    ) -> (Range<RegionVid>, Vec<RegionVariableOrigin>) {
        let range = self.unification_table.vars_since_snapshot(&mark.region_snapshot);
        (range.clone(), (range.start.index()..range.end.index()).map(|index| {
            self.var_infos[ty::RegionVid::from(index)].origin.clone()
        }).collect())
    }

    /// See [`RegionInference::region_constraints_added_in_snapshot`].
    pub fn region_constraints_added_in_snapshot(&self, mark: &RegionSnapshot) -> Option<bool> {
        self.undo_log[mark.length..]
            .iter()
            .map(|&elt| match elt {
                AddConstraint(constraint) => Some(constraint.involves_placeholders()),
                _ => None,
            }).max()
            .unwrap_or(None)
    }
}

impl fmt::Debug for RegionSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RegionSnapshot(length={})", self.length)
    }
}

impl<'tcx> fmt::Debug for GenericKind<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            GenericKind::Param(ref p) => write!(f, "{:?}", p),
            GenericKind::Projection(ref p) => write!(f, "{:?}", p),
        }
    }
}

impl<'tcx> fmt::Display for GenericKind<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            GenericKind::Param(ref p) => write!(f, "{}", p),
            GenericKind::Projection(ref p) => write!(f, "{}", p),
        }
    }
}

impl<'tcx> GenericKind<'tcx> {
    pub fn to_ty(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match *self {
            GenericKind::Param(ref p) => p.to_ty(tcx),
            GenericKind::Projection(ref p) => tcx.mk_projection(p.item_def_id, p.substs),
        }
    }
}

impl<'tcx> VerifyBound<'tcx> {
    pub fn must_hold(&self) -> bool {
        match self {
            VerifyBound::IfEq(..) => false,
            VerifyBound::OutlivedBy(ty::ReStatic) => true,
            VerifyBound::OutlivedBy(_) => false,
            VerifyBound::AnyBound(bs) => bs.iter().any(|b| b.must_hold()),
            VerifyBound::AllBounds(bs) => bs.iter().all(|b| b.must_hold()),
        }
    }

    pub fn cannot_hold(&self) -> bool {
        match self {
            VerifyBound::IfEq(_, b) => b.cannot_hold(),
            VerifyBound::OutlivedBy(ty::ReEmpty) => true,
            VerifyBound::OutlivedBy(_) => false,
            VerifyBound::AnyBound(bs) => bs.iter().all(|b| b.cannot_hold()),
            VerifyBound::AllBounds(bs) => bs.iter().any(|b| b.cannot_hold()),
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

impl<'tcx> RegionConstraintData<'tcx> {
    /// Returns `true` if this region constraint data contains no constraints, and `false`
    /// otherwise.
    pub fn is_empty(&self) -> bool {
        let RegionConstraintData {
            constraints,
            member_constraints,
            verifys,
            givens,
        } = self;
        constraints.is_empty() &&
            member_constraints.is_empty() &&
            verifys.is_empty() &&
            givens.is_empty()
    }
}
