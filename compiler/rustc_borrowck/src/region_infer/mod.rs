use std::rc::Rc;

use rustc_data_structures::frozen::Frozen;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::graph::scc::Sccs;
use rustc_index::IndexVec;
use rustc_infer::infer::outlives::test_type_match;
use rustc_infer::infer::region_constraints::{GenericKind, VerifyBound, VerifyIfEq};
use rustc_infer::infer::{InferCtxt, NllRegionVariableOrigin};
use rustc_middle::bug;
use rustc_middle::mir::{BasicBlock, Body, ConstraintCategory, Local, Location, TerminatorKind};
use rustc_middle::ty::{self, RegionVid, Ty, TyCtxt, TypeFoldable, UniverseIndex, fold_regions};
use rustc_mir_dataflow::points::DenseLocationMap;
use rustc_span::Span;
use tracing::{Level, debug, enabled, instrument};

use crate::constraints::{ConstraintSccIndex, OutlivesConstraintSet};
use crate::diagnostics::{RegionErrorKind, RegionErrors};
use crate::handle_placeholders::{RegionDefinitions, RegionTracker};
use crate::polonius::legacy::PoloniusOutput;
use crate::region_infer::universal_regions::UniversalRegionChecker;
use crate::region_infer::values::{
    LivenessValues, PlaceholderIndices, RegionValues, ToElementIndex,
};
use crate::type_check::Locations;
use crate::type_check::free_region_relations::UniversalRegionRelations;
use crate::universal_regions::UniversalRegions;
use crate::{
    BorrowckInferCtxt, ClosureOutlivesRequirement, ClosureOutlivesSubject,
    ClosureOutlivesSubjectTy, ClosureRegionRequirements,
};

mod dump_mir;
pub(crate) mod graphviz;
pub(crate) mod opaque_types;
mod reverse_sccs;

pub(crate) mod values;
pub(crate) use dump_mir::MirDumper;
mod constraint_search;
mod universal_regions;
pub(crate) use constraint_search::{BlameConstraint, ConstraintSearch};

/// The representative region variable for an SCC, tagged by its origin.
/// We prefer placeholders over existentially quantified variables, otherwise
/// it's the one with the smallest Region Variable ID. In other words,
/// the order of this enumeration really matters!
#[derive(Copy, Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub(crate) enum Representative {
    FreeRegion(RegionVid),
    Placeholder(RegionVid),
    Existential(RegionVid),
}

impl Representative {
    pub(crate) fn rvid(self) -> RegionVid {
        match self {
            Representative::FreeRegion(region_vid)
            | Representative::Placeholder(region_vid)
            | Representative::Existential(region_vid) => region_vid,
        }
    }

    pub(crate) fn new(r: RegionVid, definition: &RegionDefinition<'_>) -> Self {
        match definition.origin {
            NllRegionVariableOrigin::FreeRegion => Representative::FreeRegion(r),
            NllRegionVariableOrigin::Placeholder(_) => Representative::Placeholder(r),
            NllRegionVariableOrigin::Existential { .. } => Representative::Existential(r),
        }
    }
}

pub(crate) type ConstraintSccs = Sccs<RegionVid, ConstraintSccIndex>;

pub struct InferredRegions<'tcx> {
    pub(crate) scc_values: RegionValues<'tcx, ConstraintSccIndex>,
    pub(crate) sccs: ConstraintSccs,
    annotations: IndexVec<ConstraintSccIndex, RegionTracker>,
    universal_region_relations: Rc<Frozen<UniversalRegionRelations<'tcx>>>,
}

impl<'tcx> InferredRegions<'tcx> {
    /// Tries to find the terminator of the loop in which the region 'r' resides.
    /// Returns the location of the terminator if found.
    pub(crate) fn find_loop_terminator_location(
        &self,
        r: RegionVid,
        body: &Body<'_>,
    ) -> Option<Location> {
        let locations = self.scc_values.locations_outlived_by(self.scc(r));
        for location in locations {
            let bb = &body[location.block];
            if let Some(terminator) = &bb.terminator
                // terminator of a loop should be TerminatorKind::FalseUnwind
                && let TerminatorKind::FalseUnwind { .. } = terminator.kind
            {
                return Some(location);
            }
        }
        None
    }

    pub(crate) fn scc(&self, r: RegionVid) -> ConstraintSccIndex {
        self.sccs.scc(r)
    }

    /// Returns the lowest statement index in `start..=end` which is not contained by `r`.
    pub(crate) fn first_non_contained_inclusive(
        &self,
        r: RegionVid,
        block: BasicBlock,
        start: usize,
        end: usize,
    ) -> Option<usize> {
        self.scc_values.first_non_contained_inclusive(self.scc(r), block, start, end)
    }

    /// Returns `true` if the region `r` contains the point `p`.
    pub(crate) fn region_contains(&self, r: RegionVid, p: impl ToElementIndex<'tcx>) -> bool {
        self.scc_values.contains(self.scc(r), p)
    }

    /// Returns access to the value of `r` for debugging purposes.
    pub(crate) fn region_value_str(&self, r: RegionVid) -> String {
        self.scc_values.region_value_str(self.scc(r))
    }

    pub(crate) fn placeholders_contained_in(
        &self,
        r: RegionVid,
    ) -> impl Iterator<Item = ty::PlaceholderRegion<'tcx>> {
        self.scc_values.placeholders_contained_in(self.scc(r))
    }

    /// Check if the SCC of `r` contains `upper`.
    pub(crate) fn upper_bound_in_region_scc(&self, r: RegionVid, upper: RegionVid) -> bool {
        self.scc_values.contains(self.scc(r), upper)
    }

    pub(crate) fn universal_regions_outlived_by(
        &self,
        r: RegionVid,
    ) -> impl Iterator<Item = RegionVid> {
        self.scc_values.universal_regions_outlived_by(self.scc(r))
    }

    fn max_nameable_universe(&self, vid: RegionVid) -> UniverseIndex {
        self.annotations[self.scc(vid)].max_nameable_universe()
    }

    /// Evaluate whether `sup_region == sub_region`.
    ///
    // This is `pub` because it's used by unstable external borrowck data users, see `consumers.rs`.
    pub fn eval_equal(&self, r1: RegionVid, r2: RegionVid) -> bool {
        self.eval_outlives(r1, r2) && self.eval_outlives(r2, r1)
    }

    /// Evaluate whether `sup_region: sub_region`.
    ///
    // This is `pub` because it's used by unstable external borrowck data users, see `consumers.rs`.
    #[instrument(skip(self), level = "debug", ret)]
    pub fn eval_outlives(&self, sup_region: RegionVid, sub_region: RegionVid) -> bool {
        let sub_region_scc = self.scc(sub_region);
        let sup_region_scc = self.scc(sup_region);

        if sub_region_scc == sup_region_scc {
            debug!("{sup_region:?}: {sub_region:?} holds trivially; they are in the same SCC");
            return true;
        }

        let fr_static = self.universal_region_relations.universal_regions.fr_static;

        // If we are checking that `'sup: 'sub`, and `'sub` contains
        // some placeholder that `'sup` cannot name, then this is only
        // true if `'sup` outlives static.
        //
        // Avoid infinite recursion if `sub_region` is already `'static`
        if sub_region != fr_static
            && !self.annotations[sup_region_scc]
                .can_name_all_placeholders(self.annotations[sub_region_scc])
        {
            debug!(
                "sub universe `{sub_region_scc:?}` is not nameable \
                by super `{sup_region_scc:?}`, promoting to static",
            );

            return self.eval_outlives(sup_region, fr_static);
        }

        // Both the `sub_region` and `sup_region` consist of the union
        // of some number of universal regions (along with the union
        // of various points in the CFG; ignore those points for
        // now). Therefore, the sup-region outlives the sub-region if,
        // for each universal region R1 in the sub-region, there
        // exists some region R2 in the sup-region that outlives R1.
        let universal_outlives =
            self.scc_values.universal_regions_outlived_by(sub_region_scc).all(|r1| {
                self.universal_regions_outlived_by(sup_region)
                    .any(|r2| self.universal_region_relations.outlives(r2, r1))
            });

        if !universal_outlives {
            debug!("sub region contains a universal region not present in super");
            return false;
        }

        // Now we have to compare all the points in the sub region and make
        // sure they exist in the sup region.

        if self.universal_regions().is_universal_region(sup_region) {
            // Micro-opt: universal regions contain all points.
            debug!("super is universal and hence contains all points");
            return true;
        }

        debug!("comparison between points in sup/sub");

        self.scc_values.contains_points(sup_region_scc, sub_region_scc)
    }

    fn eval_if_eq(
        &self,
        infcx: &InferCtxt<'tcx>,
        generic_ty: Ty<'tcx>,
        lower_bound: RegionVid,
        verify_if_eq_b: ty::Binder<'tcx, VerifyIfEq<'tcx>>,
    ) -> bool {
        let generic_ty = self.normalize_to_scc_representatives(infcx.tcx, generic_ty);
        let verify_if_eq_b = self.normalize_to_scc_representatives(infcx.tcx, verify_if_eq_b);
        match test_type_match::extract_verify_if_eq(infcx.tcx, &verify_if_eq_b, generic_ty) {
            Some(r) => {
                let r_vid = self.to_region_vid(r);
                self.eval_outlives(r_vid, lower_bound)
            }
            None => false,
        }
    }

    /// This is a conservative normalization procedure. It takes every
    /// free region in `value` and replaces it with the
    /// "representative" of its SCC (see `scc_representatives` field).
    /// We are guaranteed that if two values normalize to the same
    /// thing, then they are equal; this is a conservative check in
    /// that they could still be equal even if they normalize to
    /// different results. (For example, there might be two regions
    /// with the same value that are not in the same SCC).
    ///
    /// N.B., this is not an ideal approach and I would like to revisit
    /// it. However, it works pretty well in practice. In particular,
    /// this is needed to deal with projection outlives bounds like
    ///
    /// ```text
    /// <T as Foo<'0>>::Item: '1
    /// ```
    ///
    /// In particular, this routine winds up being important when
    /// there are bounds like `where <T as Foo<'a>>::Item: 'b` in the
    /// environment. In this case, if we can show that `'0 == 'a`,
    /// and that `'b: '1`, then we know that the clause is
    /// satisfied. In such cases, particularly due to limitations of
    /// the trait solver =), we usually wind up with a where-clause like
    /// `T: Foo<'a>` in scope, which thus forces `'0 == 'a` to be added as
    /// a constraint, and thus ensures that they are in the same SCC.
    ///
    /// So why can't we do a more correct routine? Well, we could
    /// *almost* use the `relate_tys` code, but the way it is
    /// currently setup it creates inference variables to deal with
    /// higher-ranked things and so forth, and right now the inference
    /// context is not permitted to make more inference variables. So
    /// we use this kind of hacky solution.
    fn normalize_to_scc_representatives<T>(&self, tcx: TyCtxt<'tcx>, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        fold_regions(tcx, value, |r, _db| {
            ty::Region::new_var(tcx, self.to_representative(self.to_region_vid(r)))
        })
    }

    /// Given a universal region in scope on the MIR, returns the
    /// corresponding index.
    ///
    /// Panics if `r` is not a registered universal region, most notably
    /// if it is a placeholder. Handling placeholders requires access to the
    /// `MirTypeckRegionConstraints`.
    fn to_region_vid(&self, r: ty::Region<'tcx>) -> RegionVid {
        self.universal_regions().to_region_vid(r)
    }

    fn universal_regions(&self) -> &UniversalRegions<'tcx> {
        &self.universal_region_relations.universal_regions
    }

    /// Tests if `test` is true when applied to `lower_bound` at
    /// `point`.
    fn eval_verify_bound(
        &self,
        infcx: &InferCtxt<'tcx>,
        generic_ty: Ty<'tcx>,
        lower_bound: RegionVid,
        verify_bound: &VerifyBound<'tcx>,
    ) -> bool {
        debug!("eval_verify_bound(lower_bound={:?}, verify_bound={:?})", lower_bound, verify_bound);

        match verify_bound {
            VerifyBound::IfEq(verify_if_eq_b) => {
                self.eval_if_eq(infcx, generic_ty, lower_bound, *verify_if_eq_b)
            }

            VerifyBound::IsEmpty => {
                self.scc_values.elements_contained_in(self.scc(lower_bound)).next().is_none()
            }

            VerifyBound::OutlivedBy(r) => {
                let r_vid = self.to_region_vid(*r);
                self.eval_outlives(r_vid, lower_bound)
            }

            VerifyBound::AnyBound(verify_bounds) => verify_bounds.iter().any(|verify_bound| {
                self.eval_verify_bound(infcx, generic_ty, lower_bound, verify_bound)
            }),

            VerifyBound::AllBounds(verify_bounds) => verify_bounds.iter().all(|verify_bound| {
                self.eval_verify_bound(infcx, generic_ty, lower_bound, verify_bound)
            }),
        }
    }

    /// This method is used to see whether the "type tests"
    /// produced by typeck were satisfied; type tests encode
    /// type-outlives relationships like `T: 'a`. See `TypeTest`
    /// for more details.
    fn check_type_tests(
        &self,
        infcx: &InferCtxt<'tcx>,
        mut propagated_outlives_requirements: Option<&mut Vec<ClosureOutlivesRequirement<'tcx>>>,
        errors_buffer: &mut RegionErrors<'tcx>,
        type_tests: &[TypeTest<'tcx>],
    ) {
        let tcx = infcx.tcx;

        // Sometimes we register equivalent type-tests that would
        // result in basically the exact same error being reported to
        // the user. Avoid that.
        let mut deduplicate_errors = FxIndexSet::default();

        for type_test in type_tests {
            debug!("check_type_test: {:?}", type_test);

            let generic_ty = type_test.generic_kind.to_ty(tcx);
            if self.eval_verify_bound(
                infcx,
                generic_ty,
                type_test.lower_bound,
                &type_test.verify_bound,
            ) {
                continue;
            }

            if let Some(propagated_outlives_requirements) = &mut propagated_outlives_requirements
                && self.try_promote_type_test(infcx, &type_test, propagated_outlives_requirements)
            {
                continue;
            }

            // Type-test failed. Report the error.
            let erased_generic_kind = infcx.tcx.erase_and_anonymize_regions(type_test.generic_kind);

            // Skip duplicate-ish errors.
            if deduplicate_errors.insert((
                erased_generic_kind,
                type_test.lower_bound,
                type_test.span,
            )) {
                debug!(
                    "check_type_test: reporting error for erased_generic_kind={:?}, \
                     lower_bound_region={:?}, \
                     type_test.span={:?}",
                    erased_generic_kind, type_test.lower_bound, type_test.span,
                );

                errors_buffer.push(RegionErrorKind::TypeTestError { type_test: type_test.clone() });
            }
        }
    }

    /// Invoked when we have some type-test (e.g., `T: 'X`) that we cannot
    /// prove to be satisfied. If this is a closure, we will attempt to
    /// "promote" this type-test into our `ClosureRegionRequirements` and
    /// hence pass it up the creator. To do this, we have to phrase the
    /// type-test in terms of external free regions, as local free
    /// regions are not nameable by the closure's creator.
    ///
    /// Promotion works as follows: we first check that the type `T`
    /// contains only regions that the creator knows about. If this is
    /// true, then -- as a consequence -- we know that all regions in
    /// the type `T` are free regions that outlive the closure body. If
    /// false, then promotion fails.
    ///
    /// Once we've promoted T, we have to "promote" `'X` to some region
    /// that is "external" to the closure. Generally speaking, a region
    /// may be the union of some points in the closure body as well as
    /// various free lifetimes. We can ignore the points in the closure
    /// body: if the type T can be expressed in terms of external regions,
    /// we know it outlives the points in the closure body. That
    /// just leaves the free regions.
    ///
    /// The idea then is to lower the `T: 'X` constraint into multiple
    /// bounds -- e.g., if `'X` is the union of two free lifetimes,
    /// `'1` and `'2`, then we would create `T: '1` and `T: '2`.
    #[instrument(level = "debug", skip(self, infcx, propagated_outlives_requirements))]
    fn try_promote_type_test(
        &self,
        infcx: &InferCtxt<'tcx>,
        type_test: &TypeTest<'tcx>,
        propagated_outlives_requirements: &mut Vec<ClosureOutlivesRequirement<'tcx>>,
    ) -> bool {
        let tcx = infcx.tcx;
        let TypeTest { generic_kind, lower_bound, span: blame_span, verify_bound: _ } = *type_test;

        let generic_ty = generic_kind.to_ty(tcx);
        let Some(subject) = self.try_promote_type_test_subject(infcx, generic_ty) else {
            return false;
        };

        debug!(
            "lower_bound = {:?} r_scc={:?} universe={:?}",
            lower_bound,
            self.scc(lower_bound),
            self.max_nameable_universe(lower_bound)
        );
        // If the type test requires that `T: 'a` where `'a` is a
        // placeholder from another universe, that effectively requires
        // `T: 'static`, so we have to propagate that requirement.
        //
        // It doesn't matter *what* universe because the promoted `T` will
        // always be in the root universe.
        if let Some(p) = self.placeholders_contained_in(lower_bound).next() {
            debug!("encountered placeholder in higher universe: {:?}, requiring 'static", p);
            let static_r = self.universal_regions().fr_static;
            propagated_outlives_requirements.push(ClosureOutlivesRequirement {
                subject,
                outlived_free_region: static_r,
                blame_span,
                category: ConstraintCategory::Boring,
            });

            // we can return here -- the code below might push add'l constraints
            // but they would all be weaker than this one.
            return true;
        }

        // For each region outlived by lower_bound find a non-local,
        // universal region (it may be the same region) and add it to
        // `ClosureOutlivesRequirement`.
        let mut found_outlived_universal_region = false;
        for ur in self.universal_regions_outlived_by(lower_bound) {
            found_outlived_universal_region = true;
            debug!("universal_region_outlived_by ur={:?}", ur);
            let non_local_ub = self.universal_region_relations.non_local_upper_bounds(ur);
            debug!(?non_local_ub);

            // This is slightly too conservative. To show T: '1, given `'2: '1`
            // and `'3: '1` we only need to prove that T: '2 *or* T: '3, but to
            // avoid potential non-determinism we approximate this by requiring
            // T: '1 and T: '2.
            for upper_bound in non_local_ub {
                debug_assert!(self.universal_regions().is_universal_region(upper_bound));
                debug_assert!(!self.universal_regions().is_local_free_region(upper_bound));

                let requirement = ClosureOutlivesRequirement {
                    subject,
                    outlived_free_region: upper_bound,
                    blame_span,
                    category: ConstraintCategory::Boring,
                };
                debug!(?requirement, "adding closure requirement");
                propagated_outlives_requirements.push(requirement);
            }
        }
        // If we succeed to promote the subject, i.e. it only contains non-local regions,
        // and fail to prove the type test inside of the closure, the `lower_bound` has to
        // also be at least as large as some universal region, as the type test is otherwise
        // trivial.
        assert!(found_outlived_universal_region);
        true
    }

    /// When we promote a type test `T: 'r`, we have to replace all region
    /// variables in the type `T` with an equal universal region from the
    /// closure signature.
    /// This is not always possible, so this is a fallible process.
    #[instrument(level = "debug", skip(self, infcx), ret)]
    fn try_promote_type_test_subject(
        &self,
        infcx: &InferCtxt<'tcx>,
        ty: Ty<'tcx>,
    ) -> Option<ClosureOutlivesSubject<'tcx>> {
        let tcx = infcx.tcx;
        let mut failed = false;
        let ty = fold_regions(tcx, ty, |r, _depth| {
            let r_vid = self.to_region_vid(r);

            // The challenge is this. We have some region variable `r`
            // whose value is a set of CFG points and universal
            // regions. We want to find if that set is *equivalent* to
            // any of the named regions found in the closure.
            // To do so, we simply check every candidate `u_r` for equality.
            self.universal_regions_outlived_by(r_vid)
                .filter(|&u_r| !self.universal_regions().is_local_free_region(u_r))
                .find(|&u_r| self.eval_equal(u_r, r_vid))
                .map(|u_r| ty::Region::new_var(tcx, u_r))
                // In case we could not find a named region to map to,
                // we will return `None` below.
                .unwrap_or_else(|| {
                    failed = true;
                    r
                })
        });

        debug!("try_promote_type_test_subject: folded ty = {:?}", ty);

        // This will be true if we failed to promote some region.
        if failed {
            return None;
        }

        Some(ClosureOutlivesSubject::Ty(ClosureOutlivesSubjectTy::bind(tcx, ty)))
    }

    /// Returns the representative `RegionVid` for a given region's SCC.
    /// See `RegionTracker` for how a region variable ID is chosen.
    ///
    /// It is a hacky way to manage checking regions for equality,
    /// since we can 'canonicalize' each region to the representative
    /// of its SCC and be sure that -- if they have the same repr --
    /// they *must* be equal (though not having the same repr does not
    /// mean they are unequal).
    fn to_representative(&self, r: RegionVid) -> RegionVid {
        self.annotations[self.scc(r)].representative.rvid()
    }
}

pub(crate) struct RegionInferenceContext<'a, 'tcx> {
    /// Contains the definition for every region variable. Region
    /// variables are identified by their index (`RegionVid`). The
    /// definition contains information about where the region came
    /// from as well as its final inferred value.
    definitions: &'a RegionDefinitions<'tcx>,

    /// The liveness constraints added to each region. For most
    /// regions, these start out empty and steadily grow, though for
    /// each universally quantified region R they start out containing
    /// the entire CFG and `end(R)`.
    pub(crate) liveness_constraints: &'a mut LivenessValues,

    /// The outlives constraints computed by the type-check.
    pub(crate) constraints: &'a OutlivesConstraintSet<'tcx>,

    /// The SCC computed from `constraints` and the constraint
    /// graph. We have an edge from SCC A to SCC B if `A: B`. Used to
    /// compute the values of each region.
    constraint_sccs: ConstraintSccs,

    scc_annotations: IndexVec<ConstraintSccIndex, RegionTracker>,

    /// Information about how the universally quantified regions in
    /// scope on this function relate to one another.
    universal_region_relations: Rc<Frozen<UniversalRegionRelations<'tcx>>>,
}

#[derive(Debug)]
pub(crate) struct RegionDefinition<'tcx> {
    /// What kind of variable is this -- a free region? existential
    /// variable? etc. (See the `NllRegionVariableOrigin` for more
    /// info.)
    pub(crate) origin: NllRegionVariableOrigin<'tcx>,

    /// Which universe is this region variable defined in? This is
    /// most often `ty::UniverseIndex::ROOT`, but when we encounter
    /// forall-quantifiers like `for<'a> { 'a = 'b }`, we would create
    /// the variable for `'a` in a fresh universe that extends ROOT.
    pub(crate) universe: ty::UniverseIndex,

    /// If this is 'static or an early-bound region, then this is
    /// `Some(X)` where `X` is the name of the region.
    pub(crate) external_name: Option<ty::Region<'tcx>>,
}

/// N.B., the variants in `Cause` are intentionally ordered. Lower
/// values are preferred when it comes to error messages. Do not
/// reorder willy nilly.
#[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub(crate) enum Cause {
    /// point inserted because Local was live at the given Location
    LiveVar(Local, Location),

    /// point inserted because Local was dropped at the given Location
    DropVar(Local, Location),
}

/// A "type test" corresponds to an outlives constraint between a type
/// and a lifetime, like `T: 'x` or `<T as Foo>::Bar: 'x`. They are
/// translated from the `Verify` region constraints in the ordinary
/// inference context.
///
/// These sorts of constraints are handled differently than ordinary
/// constraints, at least at present. During type checking, the
/// `InferCtxt::process_registered_region_obligations` method will
/// attempt to convert a type test like `T: 'x` into an ordinary
/// outlives constraint when possible (for example, `&'a T: 'b` will
/// be converted into `'a: 'b` and registered as a `Constraint`).
///
/// In some cases, however, there are outlives relationships that are
/// not converted into a region constraint, but rather into one of
/// these "type tests". The distinction is that a type test does not
/// influence the inference result, but instead just examines the
/// values that we ultimately inferred for each region variable and
/// checks that they meet certain extra criteria. If not, an error
/// can be issued.
///
/// One reason for this is that these type tests typically boil down
/// to a check like `'a: 'x` where `'a` is a universally quantified
/// region -- and therefore not one whose value is really meant to be
/// *inferred*, precisely (this is not always the case: one can have a
/// type test like `<Foo as Trait<'?0>>::Bar: 'x`, where `'?0` is an
/// inference variable). Another reason is that these type tests can
/// involve *disjunction* -- that is, they can be satisfied in more
/// than one way.
///
/// For more information about this translation, see
/// `InferCtxt::process_registered_region_obligations` and
/// `InferCtxt::type_must_outlive` in `rustc_infer::infer::InferCtxt`.
#[derive(Clone, Debug)]
pub(crate) struct TypeTest<'tcx> {
    /// The type `T` that must outlive the region.
    pub generic_kind: GenericKind<'tcx>,

    /// The region `'x` that the type must outlive.
    pub lower_bound: RegionVid,

    /// The span to blame.
    pub span: Span,

    /// A test which, if met by the region `'x`, proves that this type
    /// constraint is satisfied.
    pub verify_bound: VerifyBound<'tcx>,
}

#[instrument(skip(infcx, sccs), level = "debug")]
fn sccs_info<'tcx>(infcx: &BorrowckInferCtxt<'tcx>, sccs: &ConstraintSccs) {
    use crate::renumber::RegionCtxt;

    let var_to_origin = infcx.reg_var_to_origin.borrow();

    let mut var_to_origin_sorted = var_to_origin.clone().into_iter().collect::<Vec<_>>();
    var_to_origin_sorted.sort_by_key(|vto| vto.0);

    if enabled!(Level::DEBUG) {
        let mut reg_vars_to_origins_str = "region variables to origins:\n".to_string();
        for (reg_var, origin) in var_to_origin_sorted.into_iter() {
            reg_vars_to_origins_str.push_str(&format!("{reg_var:?}: {origin:?}\n"));
        }
        debug!("{}", reg_vars_to_origins_str);
    }

    let num_components = sccs.num_sccs();
    let mut components = vec![FxIndexSet::default(); num_components];

    for (reg_var, scc_idx) in sccs.scc_indices().iter_enumerated() {
        let origin = var_to_origin.get(&reg_var).unwrap_or(&RegionCtxt::Unknown);
        components[scc_idx.as_usize()].insert((reg_var, *origin));
    }

    if enabled!(Level::DEBUG) {
        let mut components_str = "strongly connected components:".to_string();
        for (scc_idx, reg_vars_origins) in components.iter().enumerate() {
            let regions_info = reg_vars_origins.clone().into_iter().collect::<Vec<_>>();
            components_str.push_str(&format!(
                "{:?}: {:?},\n)",
                ConstraintSccIndex::from_usize(scc_idx),
                regions_info,
            ))
        }
        debug!("{}", components_str);
    }

    // calculate the best representative for each component
    let components_representatives = components
        .into_iter()
        .enumerate()
        .map(|(scc_idx, region_ctxts)| {
            let repr = region_ctxts
                .into_iter()
                .map(|reg_var_origin| reg_var_origin.1)
                .max_by(|x, y| x.preference_value().cmp(&y.preference_value()))
                .unwrap();

            (ConstraintSccIndex::from_usize(scc_idx), repr)
        })
        .collect::<FxIndexMap<_, _>>();

    let mut scc_node_to_edges = FxIndexMap::default();
    for (scc_idx, repr) in components_representatives.iter() {
        let edge_representatives = sccs
            .successors(*scc_idx)
            .iter()
            .map(|scc_idx| components_representatives[scc_idx])
            .collect::<Vec<_>>();
        scc_node_to_edges.insert((scc_idx, repr), edge_representatives);
    }

    debug!("SCC edges {:#?}", scc_node_to_edges);
}

impl<'a, 'tcx> RegionInferenceContext<'a, 'tcx> {
    /// Performs region inference and report errors if we see any
    /// unsatisfiable constraints. If this is a closure, returns the
    /// region requirements to propagate to our creator, if any.
    #[instrument(
        skip_all,
        level = "debug",
        fields(definitions, type_tests, universal_region_relations)
    )]
    pub(super) fn infer_regions(
        infcx: &BorrowckInferCtxt<'tcx>,
        constraint_sccs: Sccs<RegionVid, ConstraintSccIndex>,
        definitions: &'a RegionDefinitions<'tcx>,
        scc_annotations: IndexVec<ConstraintSccIndex, RegionTracker>,
        outlives_constraints: &'a OutlivesConstraintSet<'tcx>,
        type_tests: &'a [TypeTest<'tcx>],
        liveness_constraints: &'a mut LivenessValues,
        universal_region_relations: Rc<Frozen<UniversalRegionRelations<'tcx>>>,
        body: &Body<'tcx>,
        polonius_output: Option<Box<PoloniusOutput>>,
        location_map: Rc<DenseLocationMap>,
        placeholder_indices: PlaceholderIndices<'tcx>,
    ) -> (Option<ClosureRegionRequirements<'tcx>>, RegionErrors<'tcx>, InferredRegions<'tcx>) {
        let num_external_vids =
            universal_region_relations.universal_regions.num_global_and_external_regions();

        if cfg!(debug_assertions) {
            sccs_info(infcx, &constraint_sccs);
        }

        let regioncx = Self {
            definitions,
            liveness_constraints,
            constraints: outlives_constraints,
            constraint_sccs,
            scc_annotations,
            universal_region_relations,
        };

        let mir_def_id = body.source.def_id();
        let scc_values = InferredRegions {
            scc_values: regioncx.compute_region_values(location_map, placeholder_indices),
            sccs: regioncx.constraint_sccs,
            annotations: regioncx.scc_annotations,
            universal_region_relations: regioncx.universal_region_relations,
        };

        let mut errors_buffer = RegionErrors::new(infcx.tcx);

        // If this is a closure, we can propagate unsatisfied
        // `outlives_requirements` to our creator, so create a vector
        // to store those. Otherwise, we'll pass in `None` to the
        // functions below, which will trigger them to report errors
        // eagerly.
        let mut outlives_requirements = infcx.infcx.tcx.is_typeck_child(mir_def_id).then(Vec::new);

        scc_values.check_type_tests(
            infcx,
            outlives_requirements.as_mut(),
            &mut errors_buffer,
            type_tests,
        );

        debug!(?errors_buffer);
        debug!(?outlives_requirements);

        UniversalRegionChecker::new(
            &mut errors_buffer,
            definitions,
            outlives_constraints,
            &scc_values,
            regioncx.liveness_constraints,
        )
        .check(polonius_output, outlives_requirements.as_mut());

        debug!(?errors_buffer);

        let outlives_requirements = outlives_requirements.unwrap_or_default();

        if outlives_requirements.is_empty() {
            (None, errors_buffer, scc_values)
        } else {
            (
                Some(ClosureRegionRequirements { num_external_vids, outlives_requirements }),
                errors_buffer,
                scc_values,
            )
        }
    }

    /// Propagate the region constraints: this will grow the values
    /// for each region variable until all the constraints are
    /// satisfied. Note that some values may grow **too** large to be
    /// feasible, but we check this later.
    #[instrument(skip(self, location_map, placeholder_indices), level = "debug")]
    fn compute_region_values(
        &self,
        location_map: Rc<DenseLocationMap>,
        placeholder_indices: PlaceholderIndices<'tcx>,
    ) -> RegionValues<'tcx, ConstraintSccIndex> {
        debug!("constraints={:#?}", {
            let mut constraints: Vec<_> = self.constraints.outlives().iter().collect();
            constraints.sort_by_key(|c| (c.sup, c.sub));
            constraints
                .into_iter()
                .map(|c| (c, self.constraint_sccs.scc(c.sup), self.constraint_sccs.scc(c.sub)))
                .collect::<Vec<_>>()
        });

        let mut scc_values =
            RegionValues::new(location_map, self.universal_regions().len(), placeholder_indices);

        for region in self.liveness_constraints.regions() {
            scc_values.merge_liveness(self.scc(region), region, &self.liveness_constraints);
        }

        for variable in self.definitions.indices() {
            match self.definitions[variable].origin {
                NllRegionVariableOrigin::FreeRegion => {
                    // For each free, universally quantified region X:
                    scc_values.add_all_points(self.scc(variable));

                    // Add `end(X)` into the set for X.
                    scc_values.add_element(self.scc(variable), variable);
                }

                NllRegionVariableOrigin::Placeholder(placeholder) => {
                    scc_values.add_element(self.scc(variable), placeholder);
                }

                NllRegionVariableOrigin::Existential { .. } => {
                    // For existential, regions, nothing to do.
                }
            }
        }

        // To propagate constraints, we walk the DAG induced by the
        // SCC. For each SCC `A`, we visit its successors and compute
        // their values, then we union all those values to get our
        // own.
        for scc_a in self.constraint_sccs.all_sccs() {
            // Walk each SCC `B` such that `A: B`...
            for &scc_b in self.constraint_sccs.successors(scc_a) {
                debug!(?scc_b);
                scc_values.add_region(scc_a, scc_b);
            }
        }
        scc_values
    }

    fn universal_regions(&self) -> &UniversalRegions<'tcx> {
        &self.universal_region_relations.universal_regions
    }

    fn scc(&self, r: RegionVid) -> ConstraintSccIndex {
        self.constraint_sccs.scc(r)
    }
}
