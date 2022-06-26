//! See `README.md`.

use self::CombineMapType::*;
use self::UndoLog::*;

use super::{
    InferCtxtUndoLogs, MiscVariable, RegionVariableOrigin, Rollback, Snapshot, SubregionOrigin,
};

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::intern::Interned;
use rustc_data_structures::sync::Lrc;
use rustc_data_structures::undo_log::UndoLogs;
use rustc_data_structures::unify as ut;
use rustc_index::vec::IndexVec;
use rustc_middle::infer::unify_key::{RegionVidKey, UnifiedRegion};
use rustc_middle::ty::ReStatic;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{ReLateBound, ReVar};
use rustc_middle::ty::{Region, RegionVid};
use rustc_span::Span;

use std::collections::BTreeMap;
use std::ops::Range;
use std::{cmp, fmt, mem};

mod leak_check;

pub use rustc_middle::infer::MemberConstraint;

#[derive(Clone, Default)]
pub struct RegionConstraintStorage<'tcx> {
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

    /// When we add a R1 == R2 constraint, we currently add (a) edges
    /// R1 <= R2 and R2 <= R1 and (b) we unify the two regions in this
    /// table. You can then call `opportunistic_resolve_var` early
    /// which will map R1 and R2 to some common region (i.e., either
    /// R1 or R2). This is important when fulfillment, dropck and other such
    /// code is iterating to a fixed point, because otherwise we sometimes
    /// would wind up with a fresh stream of region variables that have been
    /// equated but appear distinct.
    pub(super) unification_table: ut::UnificationTableStorage<RegionVidKey<'tcx>>,

    /// a flag set to true when we perform any unifications; this is used
    /// to micro-optimize `take_and_reset_data`
    any_unifications: bool,
}

pub struct RegionConstraintCollector<'a, 'tcx> {
    storage: &'a mut RegionConstraintStorage<'tcx>,
    undo_log: &'a mut InferCtxtUndoLogs<'tcx>,
}

impl<'tcx> std::ops::Deref for RegionConstraintCollector<'_, 'tcx> {
    type Target = RegionConstraintStorage<'tcx>;
    #[inline]
    fn deref(&self) -> &RegionConstraintStorage<'tcx> {
        self.storage
    }
}

impl<'tcx> std::ops::DerefMut for RegionConstraintCollector<'_, 'tcx> {
    #[inline]
    fn deref_mut(&mut self) -> &mut RegionConstraintStorage<'tcx> {
        self.storage
    }
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
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
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

#[derive(Debug, Clone)]
pub struct Verify<'tcx> {
    pub kind: GenericKind<'tcx>,
    pub origin: SubregionOrigin<'tcx>,
    pub region: Region<'tcx>,
    pub bound: VerifyBound<'tcx>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, TypeFoldable, TypeVisitable)]
pub enum GenericKind<'tcx> {
    Param(ty::ParamTy),
    Projection(ty::ProjectionTy<'tcx>),
}

/// Describes the things that some `GenericKind` value `G` is known to
/// outlive. Each variant of `VerifyBound` can be thought of as a
/// function:
/// ```ignore (pseudo-rust)
/// fn(min: Region) -> bool { .. }
/// ```
/// where `true` means that the region `min` meets that `G: min`.
/// (False means nothing.)
///
/// So, for example, if we have the type `T` and we have in scope that
/// `T: 'a` and `T: 'b`, then the verify bound might be:
/// ```ignore (pseudo-rust)
/// fn(min: Region) -> bool {
///    ('a: min) || ('b: min)
/// }
/// ```
/// This is described with an `AnyRegion('a, 'b)` node.
#[derive(Debug, Clone, TypeFoldable, TypeVisitable)]
pub enum VerifyBound<'tcx> {
    /// See [`VerifyIfEq`] docs
    IfEq(ty::Binder<'tcx, VerifyIfEq<'tcx>>),

    /// Given a region `R`, expands to the function:
    ///
    /// ```ignore (pseudo-rust)
    /// fn(min) -> bool {
    ///     R: min
    /// }
    /// ```
    ///
    /// This is used when we can establish that `G: R` -- therefore,
    /// if `R: min`, then by transitivity `G: min`.
    OutlivedBy(Region<'tcx>),

    /// Given a region `R`, true if it is `'empty`.
    IsEmpty,

    /// Given a set of bounds `B`, expands to the function:
    ///
    /// ```ignore (pseudo-rust)
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
    /// ```ignore (pseudo-rust)
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

/// This is a "conditional bound" that checks the result of inference
/// and supplies a bound if it ended up being relevant. It's used in situations
/// like this:
///
/// ```rust
/// fn foo<'a, 'b, T: SomeTrait<'a>>
/// where
///    <T as SomeTrait<'a>>::Item: 'b
/// ```
///
/// If we have an obligation like `<T as SomeTrait<'?x>>::Item: 'c`, then
/// we don't know yet whether it suffices to show that `'b: 'c`. If `'?x` winds
/// up being equal to `'a`, then the where-clauses on function applies, and
/// in that case we can show `'b: 'c`. But if `'?x` winds up being something
/// else, the bound isn't relevant.
///
/// In the [`VerifyBound`], this struct is enclosed in `Binder to account
/// for cases like
///
/// ```rust
/// where for<'a> <T as SomeTrait<'a>::Item: 'a
/// ```
///
/// The idea is that we have to find some instantiation of `'a` that can
/// make `<T as SomeTrait<'a>>::Item` equal to the final value of `G`,
/// the generic we are checking.
///
/// ```ignore (pseudo-rust)
/// fn(min) -> bool {
///     exists<'a> {
///         if G == K {
///             B(min)
///         } else {
///             false
///         }
///     }
/// }
/// ```
#[derive(Debug, Copy, Clone, TypeFoldable, TypeVisitable)]
pub struct VerifyIfEq<'tcx> {
    /// Type which must match the generic `G`
    pub ty: Ty<'tcx>,

    /// Bound that applies if `ty` is equal.
    pub bound: Region<'tcx>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub(crate) struct TwoRegions<'tcx> {
    a: Region<'tcx>,
    b: Region<'tcx>,
}

#[derive(Copy, Clone, PartialEq)]
pub(crate) enum UndoLog<'tcx> {
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
}

#[derive(Copy, Clone, PartialEq)]
pub(crate) enum CombineMapType {
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
    any_unifications: bool,
}

impl<'tcx> RegionConstraintStorage<'tcx> {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub(crate) fn with_log<'a>(
        &'a mut self,
        undo_log: &'a mut InferCtxtUndoLogs<'tcx>,
    ) -> RegionConstraintCollector<'a, 'tcx> {
        RegionConstraintCollector { storage: self, undo_log }
    }

    fn rollback_undo_entry(&mut self, undo_entry: UndoLog<'tcx>) {
        match undo_entry {
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
}

impl<'tcx> RegionConstraintCollector<'_, 'tcx> {
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
        assert!(!UndoLogs::<super::UndoLog<'_>>::in_snapshot(&self.undo_log));
        (mem::take(&mut self.storage.var_infos), mem::take(&mut self.storage.data))
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
        assert!(!UndoLogs::<super::UndoLog<'_>>::in_snapshot(&self.undo_log));

        // If you add a new field to `RegionConstraintCollector`, you
        // should think carefully about whether it needs to be cleared
        // or updated in some way.
        let RegionConstraintStorage {
            var_infos: _,
            data,
            lubs,
            glbs,
            unification_table: _,
            any_unifications,
        } = self.storage;

        // Clear the tables of (lubs, glbs), so that we will create
        // fresh regions if we do a LUB operation. As it happens,
        // LUB/GLB are not performed by the MIR type-checker, which is
        // the one that uses this method, but it's good to be correct.
        lubs.clear();
        glbs.clear();

        let data = mem::take(data);

        // Clear all unifications and recreate the variables a "now
        // un-unified" state. Note that when we unify `a` and `b`, we
        // also insert `a <= b` and a `b <= a` edges, so the
        // `RegionConstraintData` contains the relationship here.
        if *any_unifications {
            *any_unifications = false;
            self.unification_table().reset_unifications(|_| UnifiedRegion(None));
        }

        data
    }

    pub(super) fn data(&self) -> &RegionConstraintData<'tcx> {
        &self.data
    }

    pub(super) fn start_snapshot(&mut self) -> RegionSnapshot {
        debug!("RegionConstraintCollector: start_snapshot");
        RegionSnapshot { any_unifications: self.any_unifications }
    }

    pub(super) fn rollback_to(&mut self, snapshot: RegionSnapshot) {
        debug!("RegionConstraintCollector: rollback_to({:?})", snapshot);
        self.any_unifications = snapshot.any_unifications;
    }

    pub(super) fn new_region_var(
        &mut self,
        universe: ty::UniverseIndex,
        origin: RegionVariableOrigin,
    ) -> RegionVid {
        let vid = self.var_infos.push(RegionVariableInfo { origin, universe });

        let u_vid = self.unification_table().new_key(UnifiedRegion(None));
        assert_eq!(vid, u_vid.vid);
        self.undo_log.push(AddVar(vid));
        debug!("created new region variable {:?} in {:?} with origin {:?}", vid, universe, origin);
        vid
    }

    /// Returns the universe for the given variable.
    pub(super) fn var_universe(&self, vid: RegionVid) -> ty::UniverseIndex {
        self.var_infos[vid].universe
    }

    /// Returns the origin for the given variable.
    pub(super) fn var_origin(&self, vid: RegionVid) -> RegionVariableOrigin {
        self.var_infos[vid].origin
    }

    fn add_constraint(&mut self, constraint: Constraint<'tcx>, origin: SubregionOrigin<'tcx>) {
        // cannot add constraints once regions are resolved
        debug!("RegionConstraintCollector: add_constraint({:?})", constraint);

        // never overwrite an existing (constraint, origin) - only insert one if it isn't
        // present in the map yet. This prevents origins from outside the snapshot being
        // replaced with "less informative" origins e.g., during calls to `can_eq`
        let undo_log = &mut self.undo_log;
        self.storage.data.constraints.entry(constraint).or_insert_with(|| {
            undo_log.push(AddConstraint(constraint));
            origin
        });
    }

    fn add_verify(&mut self, verify: Verify<'tcx>) {
        // cannot add verifys once regions are resolved
        debug!("RegionConstraintCollector: add_verify({:?})", verify);

        // skip no-op cases known to be satisfied
        if let VerifyBound::AllBounds(ref bs) = verify.bound && bs.is_empty() {
            return;
        }

        let index = self.data.verifys.len();
        self.data.verifys.push(verify);
        self.undo_log.push(AddVerify(index));
    }

    pub(super) fn add_given(&mut self, sub: Region<'tcx>, sup: ty::RegionVid) {
        // cannot add givens once regions are resolved
        if self.data.givens.insert((sub, sup)) {
            debug!("add_given({:?} <= {:?})", sub, sup);

            self.undo_log.push(AddGiven(sub, sup));
        }
    }

    pub(super) fn make_eqregion(
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

            match (sub, sup) {
                (Region(Interned(ReVar(sub), _)), Region(Interned(ReVar(sup), _))) => {
                    debug!("make_eqregion: unifying {:?} with {:?}", sub, sup);
                    self.unification_table().union(*sub, *sup);
                    self.any_unifications = true;
                }
                (Region(Interned(ReVar(vid), _)), value)
                | (value, Region(Interned(ReVar(vid), _))) => {
                    debug!("make_eqregion: unifying {:?} with {:?}", vid, value);
                    self.unification_table().union_value(*vid, UnifiedRegion(Some(value)));
                    self.any_unifications = true;
                }
                (_, _) => {}
            }
        }
    }

    pub(super) fn member_constraint(
        &mut self,
        key: ty::OpaqueTypeKey<'tcx>,
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
            key,
            definition_span,
            hidden_ty,
            member_region,
            choice_regions: choice_regions.clone(),
        });
    }

    #[instrument(skip(self, origin), level = "debug")]
    pub(super) fn make_subregion(
        &mut self,
        origin: SubregionOrigin<'tcx>,
        sub: Region<'tcx>,
        sup: Region<'tcx>,
    ) {
        // cannot add constraints once regions are resolved
        debug!("origin = {:#?}", origin);

        match (*sub, *sup) {
            (ReLateBound(..), _) | (_, ReLateBound(..)) => {
                span_bug!(origin.span(), "cannot relate bound region: {:?} <= {:?}", sub, sup);
            }
            (_, ReStatic) => {
                // all regions are subregions of static, so we can ignore this
            }
            (ReVar(sub_id), ReVar(sup_id)) => {
                self.add_constraint(Constraint::VarSubVar(sub_id, sup_id), origin);
            }
            (_, ReVar(sup_id)) => {
                self.add_constraint(Constraint::RegSubVar(sub, sup_id), origin);
            }
            (ReVar(sub_id), _) => {
                self.add_constraint(Constraint::VarSubReg(sub_id, sup), origin);
            }
            _ => {
                self.add_constraint(Constraint::RegSubReg(sub, sup), origin);
            }
        }
    }

    pub(super) fn verify_generic_bound(
        &mut self,
        origin: SubregionOrigin<'tcx>,
        kind: GenericKind<'tcx>,
        sub: Region<'tcx>,
        bound: VerifyBound<'tcx>,
    ) {
        self.add_verify(Verify { kind, origin, region: sub, bound });
    }

    pub(super) fn lub_regions(
        &mut self,
        tcx: TyCtxt<'tcx>,
        origin: SubregionOrigin<'tcx>,
        a: Region<'tcx>,
        b: Region<'tcx>,
    ) -> Region<'tcx> {
        // cannot add constraints once regions are resolved
        debug!("RegionConstraintCollector: lub_regions({:?}, {:?})", a, b);
        if a.is_static() || b.is_static() {
            a // nothing lives longer than static
        } else if a == b {
            a // LUB(a,a) = a
        } else {
            self.combine_vars(tcx, Lub, a, b, origin)
        }
    }

    pub(super) fn glb_regions(
        &mut self,
        tcx: TyCtxt<'tcx>,
        origin: SubregionOrigin<'tcx>,
        a: Region<'tcx>,
        b: Region<'tcx>,
    ) -> Region<'tcx> {
        // cannot add constraints once regions are resolved
        debug!("RegionConstraintCollector: glb_regions({:?}, {:?})", a, b);
        if a.is_static() {
            b // static lives longer than everything else
        } else if b.is_static() {
            a // static lives longer than everything else
        } else if a == b {
            a // GLB(a,a) = a
        } else {
            self.combine_vars(tcx, Glb, a, b, origin)
        }
    }

    /// Resolves the passed RegionVid to the root RegionVid in the unification table
    pub(super) fn opportunistic_resolve_var(&mut self, rid: ty::RegionVid) -> ty::RegionVid {
        self.unification_table().find(rid).vid
    }

    /// If the Region is a `ReVar`, then resolves it either to the root value in
    /// the unification table, if it exists, or to the root `ReVar` in the table.
    /// If the Region is not a `ReVar`, just returns the Region itself.
    pub fn opportunistic_resolve_region(
        &mut self,
        tcx: TyCtxt<'tcx>,
        region: ty::Region<'tcx>,
    ) -> ty::Region<'tcx> {
        match *region {
            ty::ReVar(rid) => {
                let unified_region = self.unification_table().probe_value(rid);
                unified_region.0.unwrap_or_else(|| {
                    let root = self.unification_table().find(rid).vid;
                    tcx.reuse_or_mk_region(region, ty::ReVar(root))
                })
            }
            _ => region,
        }
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
        let vars = TwoRegions { a, b };
        if let Some(&c) = self.combine_map(t).get(&vars) {
            return tcx.mk_region(ReVar(c));
        }
        let a_universe = self.universe(a);
        let b_universe = self.universe(b);
        let c_universe = cmp::max(a_universe, b_universe);
        let c = self.new_region_var(c_universe, MiscVariable(origin.span()));
        self.combine_map(t).insert(vars, c);
        self.undo_log.push(AddCombination(t, vars));
        let new_r = tcx.mk_region(ReVar(c));
        for old_r in [a, b] {
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
            ty::ReStatic | ty::ReErased | ty::ReFree(..) | ty::ReEarlyBound(..) => {
                ty::UniverseIndex::ROOT
            }
            ty::RePlaceholder(placeholder) => placeholder.universe,
            ty::ReVar(vid) => self.var_universe(vid),
            ty::ReLateBound(..) => bug!("universe(): encountered bound region {:?}", region),
        }
    }

    pub fn vars_since_snapshot(
        &self,
        value_count: usize,
    ) -> (Range<RegionVid>, Vec<RegionVariableOrigin>) {
        let range = RegionVid::from(value_count)..RegionVid::from(self.unification_table.len());
        (
            range.clone(),
            (range.start.index()..range.end.index())
                .map(|index| self.var_infos[ty::RegionVid::from(index)].origin)
                .collect(),
        )
    }

    /// See `InferCtxt::region_constraints_added_in_snapshot`.
    pub fn region_constraints_added_in_snapshot(&self, mark: &Snapshot<'tcx>) -> Option<bool> {
        self.undo_log
            .region_constraints_in_snapshot(mark)
            .map(|&elt| match elt {
                AddConstraint(constraint) => Some(constraint.involves_placeholders()),
                _ => None,
            })
            .max()
            .unwrap_or(None)
    }

    #[inline]
    fn unification_table(&mut self) -> super::UnificationTable<'_, 'tcx, RegionVidKey<'tcx>> {
        ut::UnificationTable::with_log(&mut self.storage.unification_table, self.undo_log)
    }
}

impl fmt::Debug for RegionSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RegionSnapshot")
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
            VerifyBound::OutlivedBy(re) => re.is_static(),
            VerifyBound::IsEmpty => false,
            VerifyBound::AnyBound(bs) => bs.iter().any(|b| b.must_hold()),
            VerifyBound::AllBounds(bs) => bs.iter().all(|b| b.must_hold()),
        }
    }

    pub fn cannot_hold(&self) -> bool {
        match self {
            VerifyBound::IfEq(..) => false,
            VerifyBound::IsEmpty => false,
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
}

impl<'tcx> RegionConstraintData<'tcx> {
    /// Returns `true` if this region constraint data contains no constraints, and `false`
    /// otherwise.
    pub fn is_empty(&self) -> bool {
        let RegionConstraintData { constraints, member_constraints, verifys, givens } = self;
        constraints.is_empty()
            && member_constraints.is_empty()
            && verifys.is_empty()
            && givens.is_empty()
    }
}

impl<'tcx> Rollback<UndoLog<'tcx>> for RegionConstraintStorage<'tcx> {
    fn reverse(&mut self, undo: UndoLog<'tcx>) {
        self.rollback_undo_entry(undo)
    }
}
