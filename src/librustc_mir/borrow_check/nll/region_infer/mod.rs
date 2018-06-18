// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::universal_regions::UniversalRegions;
use borrow_check::nll::region_infer::values::ToElementIndex;
use rustc::hir::def_id::DefId;
use rustc::infer::error_reporting::nice_region_error::NiceRegionError;
use rustc::infer::region_constraints::{GenericKind, VarInfos};
use rustc::infer::InferCtxt;
use rustc::infer::NLLRegionVariableOrigin;
use rustc::infer::RegionObligation;
use rustc::infer::RegionVariableOrigin;
use rustc::infer::SubregionOrigin;
use rustc::mir::{
    ClosureOutlivesRequirement, ClosureOutlivesSubject, ClosureRegionRequirements, Local, Location,
    Mir,
};
use rustc::traits::ObligationCause;
use rustc::ty::{self, RegionVid, Ty, TypeFoldable};
use rustc::util::common::{self, ErrorReported};
use rustc_data_structures::bitvec::BitVector;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use std::fmt;
use std::rc::Rc;
use syntax::ast;
use syntax_pos::Span;

mod annotation;
mod dump_mir;
mod graphviz;
mod values;
use self::values::{RegionValueElements, RegionValues};

use super::ToRegionVid;

pub struct RegionInferenceContext<'tcx> {
    /// Contains the definition for every region variable.  Region
    /// variables are identified by their index (`RegionVid`). The
    /// definition contains information about where the region came
    /// from as well as its final inferred value.
    definitions: IndexVec<RegionVid, RegionDefinition<'tcx>>,

    /// Maps from points/universal-regions to a `RegionElementIndex`.
    elements: Rc<RegionValueElements>,

    /// The liveness constraints added to each region. For most
    /// regions, these start out empty and steadily grow, though for
    /// each universally quantified region R they start out containing
    /// the entire CFG and `end(R)`.
    liveness_constraints: RegionValues,

    /// The final inferred values of the inference variables; `None`
    /// until `solve` is invoked.
    inferred_values: Option<RegionValues>,

    /// For each variable, stores the index of the first constraint
    /// where that variable appears on the RHS. This is the start of a
    /// 'linked list' threaded by the `next` field in `Constraint`.
    ///
    /// This map is build when values are inferred.
    dependency_map: Option<IndexVec<RegionVid, Option<ConstraintIndex>>>,

    /// The constraints we have accumulated and used during solving.
    constraints: IndexVec<ConstraintIndex, OutlivesConstraint>,

    /// Type constraints that we check after solving.
    type_tests: Vec<TypeTest<'tcx>>,

    /// Information about the universally quantified regions in scope
    /// on this function and their (known) relations to one another.
    universal_regions: UniversalRegions<'tcx>,
}

struct TrackCauses(bool);

struct RegionDefinition<'tcx> {
    /// Why we created this variable. Mostly these will be
    /// `RegionVariableOrigin::NLL`, but some variables get created
    /// elsewhere in the code with other causes (e.g., instantiation
    /// late-bound-regions).
    origin: RegionVariableOrigin,

    /// True if this is a universally quantified region. This means a
    /// lifetime parameter that appears in the function signature (or,
    /// in the case of a closure, in the closure environment, which of
    /// course is also in the function signature).
    is_universal: bool,

    /// If this is 'static or an early-bound region, then this is
    /// `Some(X)` where `X` is the name of the region.
    external_name: Option<ty::Region<'tcx>>,
}

/// NB: The variants in `Cause` are intentionally ordered. Lower
/// values are preferred when it comes to error messages. Do not
/// reorder willy nilly.
#[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub(crate) enum Cause {
    /// point inserted because Local was live at the given Location
    LiveVar(Local, Location),

    /// point inserted because Local was dropped at the given Location
    DropVar(Local, Location),

    /// point inserted because the type was live at the given Location,
    /// but not as part of some local variable
    LiveOther(Location),

    /// part of the initial set of values for a universally quantified region
    UniversalRegion(RegionVid),
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OutlivesConstraint {
    // NB. The ordering here is not significant for correctness, but
    // it is for convenience. Before we dump the constraints in the
    // debugging logs, we sort them, and we'd like the "super region"
    // to be first, etc. (In particular, span should remain last.)
    /// The region SUP must outlive SUB...
    pub sup: RegionVid,

    /// Region that must be outlived.
    pub sub: RegionVid,

    /// At this location.
    pub point: Location,

    /// Later on, we thread the constraints onto a linked list
    /// grouped by their `sub` field. So if you had:
    ///
    /// Index | Constraint | Next Field
    /// ----- | ---------- | ----------
    /// 0     | `'a: 'b`   | Some(2)
    /// 1     | `'b: 'c`   | None
    /// 2     | `'c: 'b`   | None
    pub next: Option<ConstraintIndex>,

    /// Where did this constraint arise?
    pub span: Span,
}

newtype_index!(ConstraintIndex { DEBUG_FORMAT = "ConstraintIndex({})" });

/// A "type test" corresponds to an outlives constraint between a type
/// and a lifetime, like `T: 'x` or `<T as Foo>::Bar: 'x`.  They are
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
/// these "type tests".  The distinction is that a type test does not
/// influence the inference result, but instead just examines the
/// values that we ultimately inferred for each region variable and
/// checks that they meet certain extra criteria.  If not, an error
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
/// `InferCtxt::type_must_outlive` in `rustc::infer::outlives`.
#[derive(Clone, Debug)]
pub struct TypeTest<'tcx> {
    /// The type `T` that must outlive the region.
    pub generic_kind: GenericKind<'tcx>,

    /// The region `'x` that the type must outlive.
    pub lower_bound: RegionVid,

    /// The point where the outlives relation must hold.
    pub point: Location,

    /// Where did this constraint arise?
    pub span: Span,

    /// A test which, if met by the region `'x`, proves that this type
    /// constraint is satisfied.
    pub test: RegionTest,
}

/// A "test" that can be applied to some "subject region" `'x`. These are used to
/// describe type constraints. Tests do not presently affect the
/// region values that get inferred for each variable; they only
/// examine the results *after* inference.  This means they can
/// conveniently include disjuction ("a or b must be true").
#[derive(Clone, Debug)]
pub enum RegionTest {
    /// The subject region `'x` must by outlived by *some* region in
    /// the given set of regions.
    ///
    /// This test comes from e.g. a where clause like `T: 'a + 'b`,
    /// which implies that we know that `T: 'a` and that `T:
    /// 'b`. Therefore, if we are trying to prove that `T: 'x`, we can
    /// do so by showing that `'a: 'x` *or* `'b: 'x`.
    IsOutlivedByAnyRegionIn(Vec<RegionVid>),

    /// The subject region `'x` must by outlived by *all* regions in
    /// the given set of regions.
    ///
    /// This test comes from e.g. a projection type like `T = <u32 as
    /// Trait<'a, 'b>>::Foo`, which must outlive `'a` or `'b`, and
    /// maybe both. Therefore we can prove that `T: 'x` if we know
    /// that `'a: 'x` *and* `'b: 'x`.
    IsOutlivedByAllRegionsIn(Vec<RegionVid>),

    /// Any of the given tests are true.
    ///
    /// This arises from projections, for which there are multiple
    /// ways to prove an outlives relationship.
    Any(Vec<RegionTest>),

    /// All of the given tests are true.
    All(Vec<RegionTest>),
}

impl<'tcx> RegionInferenceContext<'tcx> {
    /// Creates a new region inference context with a total of
    /// `num_region_variables` valid inference variables; the first N
    /// of those will be constant regions representing the free
    /// regions defined in `universal_regions`.
    ///
    /// The `outlives_constraints` and `type_tests` are an initial set
    /// of constraints produced by the MIR type check.
    pub(crate) fn new(
        var_infos: VarInfos,
        universal_regions: UniversalRegions<'tcx>,
        mir: &Mir<'tcx>,
        outlives_constraints: Vec<OutlivesConstraint>,
        type_tests: Vec<TypeTest<'tcx>>,
    ) -> Self {
        // The `next` field should not yet have been initialized:
        debug_assert!(outlives_constraints.iter().all(|c| c.next.is_none()));

        let num_region_variables = var_infos.len();
        let num_universal_regions = universal_regions.len();

        let elements = &Rc::new(RegionValueElements::new(mir, num_universal_regions));

        // Create a RegionDefinition for each inference variable.
        let definitions = var_infos
            .into_iter()
            .map(|info| RegionDefinition::new(info.origin))
            .collect();

        let mut result = Self {
            definitions,
            elements: elements.clone(),
            liveness_constraints: RegionValues::new(elements, num_region_variables),
            inferred_values: None,
            dependency_map: None,
            constraints: IndexVec::from_raw(outlives_constraints),
            type_tests,
            universal_regions,
        };

        result.init_universal_regions();

        result
    }

    /// Initializes the region variables for each universally
    /// quantified region (lifetime parameter). The first N variables
    /// always correspond to the regions appearing in the function
    /// signature (both named and anonymous) and where clauses. This
    /// function iterates over those regions and initializes them with
    /// minimum values.
    ///
    /// For example:
    ///
    ///     fn foo<'a, 'b>(..) where 'a: 'b
    ///
    /// would initialize two variables like so:
    ///
    ///     R0 = { CFG, R0 } // 'a
    ///     R1 = { CFG, R0, R1 } // 'b
    ///
    /// Here, R0 represents `'a`, and it contains (a) the entire CFG
    /// and (b) any universally quantified regions that it outlives,
    /// which in this case is just itself. R1 (`'b`) in contrast also
    /// outlives `'a` and hence contains R0 and R1.
    fn init_universal_regions(&mut self) {
        // Update the names (if any)
        for (external_name, variable) in self.universal_regions.named_universal_regions() {
            self.definitions[variable].external_name = Some(external_name);
        }

        // For each universally quantified region X:
        for variable in self.universal_regions.universal_regions() {
            // These should be free-region variables.
            assert!(match self.definitions[variable].origin {
                RegionVariableOrigin::NLL(NLLRegionVariableOrigin::FreeRegion) => true,
                _ => false,
            });

            self.definitions[variable].is_universal = true;

            // Add all nodes in the CFG to liveness constraints
            for point_index in self.elements.all_point_indices() {
                self.liveness_constraints.add_element(
                    variable,
                    point_index,
                    &Cause::UniversalRegion(variable),
                );
            }

            // Add `end(X)` into the set for X.
            self.liveness_constraints.add_element(
                variable,
                variable,
                &Cause::UniversalRegion(variable),
            );
        }
    }

    /// Returns an iterator over all the region indices.
    pub fn regions(&self) -> impl Iterator<Item = RegionVid> {
        self.definitions.indices()
    }

    /// Given a universal region in scope on the MIR, returns the
    /// corresponding index.
    ///
    /// (Panics if `r` is not a registered universal region.)
    pub fn to_region_vid(&self, r: ty::Region<'tcx>) -> RegionVid {
        self.universal_regions.to_region_vid(r)
    }

    /// Returns true if the region `r` contains the point `p`.
    ///
    /// Panics if called before `solve()` executes,
    pub fn region_contains_point<R>(&self, r: R, p: Location) -> bool
    where
        R: ToRegionVid,
    {
        let inferred_values = self
            .inferred_values
            .as_ref()
            .expect("region values not yet inferred");
        inferred_values.contains(r.to_region_vid(), p)
    }

    /// Returns access to the value of `r` for debugging purposes.
    crate fn region_value_str(&self, r: RegionVid) -> String {
        let inferred_values = self
            .inferred_values
            .as_ref()
            .expect("region values not yet inferred");

        inferred_values.region_value_str(r)
    }

    /// Indicates that the region variable `v` is live at the point `point`.
    ///
    /// Returns `true` if this constraint is new and `false` is the
    /// constraint was already present.
    pub(super) fn add_live_point(&mut self, v: RegionVid, point: Location, cause: &Cause) -> bool {
        debug!("add_live_point({:?}, {:?})", v, point);
        assert!(self.inferred_values.is_none(), "values already inferred");
        debug!("add_live_point: @{:?} Adding cause {:?}", point, cause);

        let element = self.elements.index(point);
        if self.liveness_constraints.add_element(v, element, &cause) {
            true
        } else {
            false
        }
    }

    /// Indicates that the region variable `sup` must outlive `sub` is live at the point `point`.
    pub(super) fn add_outlives(
        &mut self,
        span: Span,
        sup: RegionVid,
        sub: RegionVid,
        point: Location,
    ) {
        debug!("add_outlives({:?}: {:?} @ {:?}", sup, sub, point);
        assert!(self.inferred_values.is_none(), "values already inferred");
        self.constraints.push(OutlivesConstraint {
            span,
            sup,
            sub,
            point,
            next: None,
        });
    }

    /// Perform region inference and report errors if we see any
    /// unsatisfiable constraints. If this is a closure, returns the
    /// region requirements to propagate to our creator, if any.
    pub(super) fn solve<'gcx>(
        &mut self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
        mir: &Mir<'tcx>,
        mir_def_id: DefId,
    ) -> Option<ClosureRegionRequirements<'gcx>> {
        common::time(
            infcx.tcx.sess,
            &format!("solve_nll_region_constraints({:?})", mir_def_id),
            || self.solve_inner(infcx, mir, mir_def_id),
        )
    }

    fn solve_inner<'gcx>(
        &mut self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
        mir: &Mir<'tcx>,
        mir_def_id: DefId,
    ) -> Option<ClosureRegionRequirements<'gcx>> {
        assert!(self.inferred_values.is_none(), "values already inferred");

        self.propagate_constraints(mir);

        // If this is a closure, we can propagate unsatisfied
        // `outlives_requirements` to our creator, so create a vector
        // to store those. Otherwise, we'll pass in `None` to the
        // functions below, which will trigger them to report errors
        // eagerly.
        let mut outlives_requirements = if infcx.tcx.is_closure(mir_def_id) {
            Some(vec![])
        } else {
            None
        };

        self.check_type_tests(infcx, mir, mir_def_id, outlives_requirements.as_mut());

        self.check_universal_regions(infcx, mir_def_id, outlives_requirements.as_mut());

        let outlives_requirements = outlives_requirements.unwrap_or(vec![]);

        if outlives_requirements.is_empty() {
            None
        } else {
            let num_external_vids = self.universal_regions.num_global_and_external_regions();
            Some(ClosureRegionRequirements {
                num_external_vids,
                outlives_requirements,
            })
        }
    }

    /// Propagate the region constraints: this will grow the values
    /// for each region variable until all the constraints are
    /// satisfied. Note that some values may grow **too** large to be
    /// feasible, but we check this later.
    fn propagate_constraints(&mut self, mir: &Mir<'tcx>) {
        self.dependency_map = Some(self.build_dependency_map());
        let inferred_values = self.compute_region_values(mir);
        self.inferred_values = Some(inferred_values);
    }

    fn compute_region_values(&self, _mir: &Mir<'tcx>) -> RegionValues {
        debug!("compute_region_values()");
        debug!("compute_region_values: constraints={:#?}", {
            let mut constraints: Vec<_> = self.constraints.iter().collect();
            constraints.sort();
            constraints
        });

        // The initial values for each region are derived from the liveness
        // constraints we have accumulated.
        let mut inferred_values = self.liveness_constraints.duplicate(TrackCauses(false));

        let dependency_map = self.dependency_map.as_ref().unwrap();

        // Constraints that may need to be repropagated (initially all):
        let mut dirty_list: Vec<_> = self.constraints.indices().collect();

        // Set to 0 for each constraint that is on the dirty list:
        let mut clean_bit_vec = BitVector::new(dirty_list.len());

        debug!("propagate_constraints: --------------------");
        while let Some(constraint_idx) = dirty_list.pop() {
            clean_bit_vec.insert(constraint_idx.index());

            let constraint = &self.constraints[constraint_idx];
            debug!("propagate_constraints: constraint={:?}", constraint);

            if inferred_values.add_region(constraint.sup, constraint.sub) {
                debug!("propagate_constraints:   sub={:?}", constraint.sub);
                debug!("propagate_constraints:   sup={:?}", constraint.sup);

                let mut opt_dep_idx = dependency_map[constraint.sup];
                while let Some(dep_idx) = opt_dep_idx {
                    if clean_bit_vec.remove(dep_idx.index()) {
                        dirty_list.push(dep_idx);
                    }
                    opt_dep_idx = self.constraints[dep_idx].next;
                }
            }

            debug!("\n");
        }

        inferred_values
    }

    /// Builds up a map from each region variable X to a vector with the
    /// indices of constraints that need to be re-evaluated when X changes.
    /// These are constraints like Y: X @ P -- so if X changed, we may
    /// need to grow Y.
    fn build_dependency_map(&mut self) -> IndexVec<RegionVid, Option<ConstraintIndex>> {
        let mut map = IndexVec::from_elem(None, &self.definitions);

        for (idx, constraint) in self.constraints.iter_enumerated_mut().rev() {
            let mut head = &mut map[constraint.sub];
            debug_assert!(constraint.next.is_none());
            constraint.next = *head;
            *head = Some(idx);
        }

        map
    }

    /// Once regions have been propagated, this method is used to see
    /// whether the "type tests" produced by typeck were satisfied;
    /// type tests encode type-outlives relationships like `T:
    /// 'a`. See `TypeTest` for more details.
    fn check_type_tests<'gcx>(
        &self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
        mir: &Mir<'tcx>,
        mir_def_id: DefId,
        mut propagated_outlives_requirements: Option<&mut Vec<ClosureOutlivesRequirement<'gcx>>>,
    ) {
        let tcx = infcx.tcx;

        for type_test in &self.type_tests {
            debug!("check_type_test: {:?}", type_test);

            if self.eval_region_test(mir, type_test.point, type_test.lower_bound, &type_test.test) {
                continue;
            }

            if let Some(propagated_outlives_requirements) = &mut propagated_outlives_requirements {
                if self.try_promote_type_test(infcx, type_test, propagated_outlives_requirements) {
                    continue;
                }
            }

            // Oh the humanity. Obviously we will do better than this error eventually.
            let lower_bound_region = self.to_error_region(type_test.lower_bound);
            if let Some(lower_bound_region) = lower_bound_region {
                let region_scope_tree = &tcx.region_scope_tree(mir_def_id);
                infcx.report_generic_bound_failure(
                    region_scope_tree,
                    type_test.span,
                    None,
                    type_test.generic_kind,
                    lower_bound_region,
                );
            } else {
                // FIXME. We should handle this case better. It
                // indicates that we have e.g. some region variable
                // whose value is like `'a+'b` where `'a` and `'b` are
                // distinct unrelated univesal regions that are not
                // known to outlive one another. It'd be nice to have
                // some examples where this arises to decide how best
                // to report it; we could probably handle it by
                // iterating over the universal regions and reporting
                // an error that multiple bounds are required.
                tcx.sess.span_err(
                    type_test.span,
                    &format!("`{}` does not live long enough", type_test.generic_kind,),
                );
            }
        }
    }

    /// Converts a region inference variable into a `ty::Region` that
    /// we can use for error reporting. If `r` is universally bound,
    /// then we use the name that we have on record for it. If `r` is
    /// existentially bound, then we check its inferred value and try
    /// to find a good name from that. Returns `None` if we can't find
    /// one (e.g., this is just some random part of the CFG).
    pub fn to_error_region(&self, r: RegionVid) -> Option<ty::Region<'tcx>> {
        if self.universal_regions.is_universal_region(r) {
            return self.definitions[r].external_name;
        } else {
            let inferred_values = self
                .inferred_values
                .as_ref()
                .expect("region values not yet inferred");
            let upper_bound = self.universal_upper_bound(r);
            if inferred_values.contains(r, upper_bound) {
                self.to_error_region(upper_bound)
            } else {
                None
            }
        }
    }

    fn try_promote_type_test<'gcx>(
        &self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
        type_test: &TypeTest<'tcx>,
        propagated_outlives_requirements: &mut Vec<ClosureOutlivesRequirement<'gcx>>,
    ) -> bool {
        let tcx = infcx.tcx;

        let TypeTest {
            generic_kind,
            lower_bound,
            point: _,
            span,
            test: _,
        } = type_test;

        let generic_ty = generic_kind.to_ty(tcx);
        let subject = match self.try_promote_type_test_subject(infcx, generic_ty) {
            Some(s) => s,
            None => return false,
        };

        // Find some bounding subject-region R+ that is a super-region
        // of the existing subject-region R. This should be a non-local, universal
        // region, which ensures it can be encoded in a `ClosureOutlivesRequirement`.
        let lower_bound_plus = self.non_local_universal_upper_bound(*lower_bound);
        assert!(self.universal_regions.is_universal_region(lower_bound_plus));
        assert!(
            !self
                .universal_regions
                .is_local_free_region(lower_bound_plus)
        );

        propagated_outlives_requirements.push(ClosureOutlivesRequirement {
            subject,
            outlived_free_region: lower_bound_plus,
            blame_span: *span,
        });
        true
    }

    /// When we promote a type test `T: 'r`, we have to convert the
    /// type `T` into something we can store in a query result (so
    /// something allocated for `'gcx`). This is problematic if `ty`
    /// contains regions. During the course of NLL region checking, we
    /// will have replaced all of those regions with fresh inference
    /// variables. To create a test subject, we want to replace those
    /// inference variables with some region from the closure
    /// signature -- this is not always possible, so this is a
    /// fallible process. Presuming we do find a suitable region, we
    /// will represent it with a `ReClosureBound`, which is a
    /// `RegionKind` variant that can be allocated in the gcx.
    fn try_promote_type_test_subject<'gcx>(
        &self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
        ty: Ty<'tcx>,
    ) -> Option<ClosureOutlivesSubject<'gcx>> {
        let tcx = infcx.tcx;
        let gcx = tcx.global_tcx();
        let inferred_values = self
            .inferred_values
            .as_ref()
            .expect("region values not yet inferred");

        debug!("try_promote_type_test_subject(ty = {:?})", ty);

        let ty = tcx.fold_regions(&ty, &mut false, |r, _depth| {
            let region_vid = self.to_region_vid(r);

            // The challenge if this. We have some region variable `r`
            // whose value is a set of CFG points and universal
            // regions. We want to find if that set is *equivalent* to
            // any of the named regions found in the closure.
            //
            // To do so, we compute the
            // `non_local_universal_upper_bound`. This will be a
            // non-local, universal region that is greater than `r`.
            // However, it might not be *contained* within `r`, so
            // then we further check whether this bound is contained
            // in `r`. If so, we can say that `r` is equivalent to the
            // bound.
            //
            // Let's work through a few examples. For these, imagine
            // that we have 3 non-local regions (I'll denote them as
            // `'static`, `'a`, and `'b`, though of course in the code
            // they would be represented with indices) where:
            //
            // - `'static: 'a`
            // - `'static: 'b`
            //
            // First, let's assume that `r` is some existential
            // variable with an inferred value `{'a, 'static}` (plus
            // some CFG nodes). In this case, the non-local upper
            // bound is `'static`, since that outlives `'a`. `'static`
            // is also a member of `r` and hence we consider `r`
            // equivalent to `'static` (and replace it with
            // `'static`).
            //
            // Now let's consider the inferred value `{'a, 'b}`. This
            // means `r` is effectively `'a | 'b`. I'm not sure if
            // this can come about, actually, but assuming it did, we
            // would get a non-local upper bound of `'static`. Since
            // `'static` is not contained in `r`, we would fail to
            // find an equivalent.
            let upper_bound = self.non_local_universal_upper_bound(region_vid);
            if inferred_values.contains(region_vid, upper_bound) {
                tcx.mk_region(ty::ReClosureBound(upper_bound))
            } else {
                // In the case of a failure, use a `ReVar`
                // result. This will cause the `lift` later on to
                // fail.
                r
            }
        });
        debug!("try_promote_type_test_subject: folded ty = {:?}", ty);

        // `lift` will only fail if we failed to promote some region.
        let ty = gcx.lift(&ty)?;

        Some(ClosureOutlivesSubject::Ty(ty))
    }

    /// Given some universal or existential region `r`, finds a
    /// non-local, universal region `r+` that outlives `r` at entry to (and
    /// exit from) the closure. In the worst case, this will be
    /// `'static`.
    ///
    /// This is used for two purposes. First, if we are propagated
    /// some requirement `T: r`, we can use this method to enlarge `r`
    /// to something we can encode for our creator (which only knows
    /// about non-local, universal regions). It is also used when
    /// encoding `T` as part of `try_promote_type_test_subject` (see
    /// that fn for details).
    ///
    /// This is based on the result `'y` of `universal_upper_bound`,
    /// except that it converts further takes the non-local upper
    /// bound of `'y`, so that the final result is non-local.
    fn non_local_universal_upper_bound(&self, r: RegionVid) -> RegionVid {
        let inferred_values = self.inferred_values.as_ref().unwrap();

        debug!(
            "non_local_universal_upper_bound(r={:?}={})",
            r,
            inferred_values.region_value_str(r)
        );

        let lub = self.universal_upper_bound(r);

        // Grow further to get smallest universal region known to
        // creator.
        let non_local_lub = self.universal_regions.non_local_upper_bound(lub);

        debug!(
            "non_local_universal_upper_bound: non_local_lub={:?}",
            non_local_lub
        );

        non_local_lub
    }

    /// Returns a universally quantified region that outlives the
    /// value of `r` (`r` may be existentially or universally
    /// quantified).
    ///
    /// Since `r` is (potentially) an existential region, it has some
    /// value which may include (a) any number of points in the CFG
    /// and (b) any number of `end('x)` elements of universally
    /// quantified regions. To convert this into a single universal
    /// region we do as follows:
    ///
    /// - Ignore the CFG points in `'r`. All universally quantified regions
    ///   include the CFG anyhow.
    /// - For each `end('x)` element in `'r`, compute the mutual LUB, yielding
    ///   a result `'y`.
    fn universal_upper_bound(&self, r: RegionVid) -> RegionVid {
        let inferred_values = self.inferred_values.as_ref().unwrap();

        debug!(
            "universal_upper_bound(r={:?}={})",
            r,
            inferred_values.region_value_str(r)
        );

        // Find the smallest universal region that contains all other
        // universal regions within `region`.
        let mut lub = self.universal_regions.fr_fn_body;
        for ur in inferred_values.universal_regions_outlived_by(r) {
            lub = self.universal_regions.postdom_upper_bound(lub, ur);
        }

        debug!("universal_upper_bound: r={:?} lub={:?}", r, lub);

        lub
    }

    /// Test if `test` is true when applied to `lower_bound` at
    /// `point`, and returns true or false.
    fn eval_region_test(
        &self,
        mir: &Mir<'tcx>,
        point: Location,
        lower_bound: RegionVid,
        test: &RegionTest,
    ) -> bool {
        debug!(
            "eval_region_test(point={:?}, lower_bound={:?}, test={:?})",
            point, lower_bound, test
        );

        match test {
            RegionTest::IsOutlivedByAllRegionsIn(regions) => regions
                .iter()
                .all(|&r| self.eval_outlives(mir, r, lower_bound, point)),

            RegionTest::IsOutlivedByAnyRegionIn(regions) => regions
                .iter()
                .any(|&r| self.eval_outlives(mir, r, lower_bound, point)),

            RegionTest::Any(tests) => tests
                .iter()
                .any(|test| self.eval_region_test(mir, point, lower_bound, test)),

            RegionTest::All(tests) => tests
                .iter()
                .all(|test| self.eval_region_test(mir, point, lower_bound, test)),
        }
    }

    // Evaluate whether `sup_region: sub_region @ point`.
    fn eval_outlives(
        &self,
        _mir: &Mir<'tcx>,
        sup_region: RegionVid,
        sub_region: RegionVid,
        point: Location,
    ) -> bool {
        debug!(
            "eval_outlives({:?}: {:?} @ {:?})",
            sup_region, sub_region, point
        );

        let inferred_values = self
            .inferred_values
            .as_ref()
            .expect("values for regions not yet inferred");

        debug!(
            "eval_outlives: sup_region's value = {:?}",
            inferred_values.region_value_str(sup_region),
        );
        debug!(
            "eval_outlives: sub_region's value = {:?}",
            inferred_values.region_value_str(sub_region),
        );

        // Both the `sub_region` and `sup_region` consist of the union
        // of some number of universal regions (along with the union
        // of various points in the CFG; ignore those points for
        // now). Therefore, the sup-region outlives the sub-region if,
        // for each universal region R1 in the sub-region, there
        // exists some region R2 in the sup-region that outlives R1.
        let universal_outlives = inferred_values
            .universal_regions_outlived_by(sub_region)
            .all(|r1| {
                inferred_values
                    .universal_regions_outlived_by(sup_region)
                    .any(|r2| self.universal_regions.outlives(r2, r1))
            });

        if !universal_outlives {
            return false;
        }

        // Now we have to compare all the points in the sub region and make
        // sure they exist in the sup region.

        if self.universal_regions.is_universal_region(sup_region) {
            // Micro-opt: universal regions contain all points.
            return true;
        }

        inferred_values.contains_points(sup_region, sub_region)
    }

    /// Once regions have been propagated, this method is used to see
    /// whether any of the constraints were too strong. In particular,
    /// we want to check for a case where a universally quantified
    /// region exceeded its bounds.  Consider:
    ///
    ///     fn foo<'a, 'b>(x: &'a u32) -> &'b u32 { x }
    ///
    /// In this case, returning `x` requires `&'a u32 <: &'b u32`
    /// and hence we establish (transitively) a constraint that
    /// `'a: 'b`. The `propagate_constraints` code above will
    /// therefore add `end('a)` into the region for `'b` -- but we
    /// have no evidence that `'b` outlives `'a`, so we want to report
    /// an error.
    ///
    /// If `propagated_outlives_requirements` is `Some`, then we will
    /// push unsatisfied obligations into there. Otherwise, we'll
    /// report them as errors.
    fn check_universal_regions<'gcx>(
        &self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
        mir_def_id: DefId,
        mut propagated_outlives_requirements: Option<&mut Vec<ClosureOutlivesRequirement<'gcx>>>,
    ) {
        // The universal regions are always found in a prefix of the
        // full list.
        let universal_definitions = self
            .definitions
            .iter_enumerated()
            .take_while(|(_, fr_definition)| fr_definition.is_universal);

        // Go through each of the universal regions `fr` and check that
        // they did not grow too large, accumulating any requirements
        // for our caller into the `outlives_requirements` vector.
        for (fr, _) in universal_definitions {
            self.check_universal_region(
                infcx,
                mir_def_id,
                fr,
                &mut propagated_outlives_requirements,
            );
        }
    }

    /// Check the final value for the free region `fr` to see if it
    /// grew too large. In particular, examine what `end(X)` points
    /// wound up in `fr`'s final value; for each `end(X)` where `X !=
    /// fr`, we want to check that `fr: X`. If not, that's either an
    /// error, or something we have to propagate to our creator.
    ///
    /// Things that are to be propagated are accumulated into the
    /// `outlives_requirements` vector.
    fn check_universal_region<'gcx>(
        &self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
        mir_def_id: DefId,
        longer_fr: RegionVid,
        propagated_outlives_requirements: &mut Option<&mut Vec<ClosureOutlivesRequirement<'gcx>>>,
    ) {
        let inferred_values = self.inferred_values.as_ref().unwrap();

        debug!("check_universal_region(fr={:?})", longer_fr);

        // Find every region `o` such that `fr: o`
        // (because `fr` includes `end(o)`).
        for shorter_fr in inferred_values.universal_regions_outlived_by(longer_fr) {
            // If it is known that `fr: o`, carry on.
            if self.universal_regions.outlives(longer_fr, shorter_fr) {
                continue;
            }

            debug!(
                "check_universal_region: fr={:?} does not outlive shorter_fr={:?}",
                longer_fr, shorter_fr,
            );

            let blame_index = self.blame_constraint(longer_fr, shorter_fr);
            let blame_span = self.constraints[blame_index].span;

            if let Some(propagated_outlives_requirements) = propagated_outlives_requirements {
                // Shrink `fr` until we find a non-local region (if we do).
                // We'll call that `fr-` -- it's ever so slightly smaller than `fr`.
                if let Some(fr_minus) = self.universal_regions.non_local_lower_bound(longer_fr) {
                    debug!("check_universal_region: fr_minus={:?}", fr_minus);

                    // Grow `shorter_fr` until we find a non-local
                    // region. (We always will.)  We'll call that
                    // `shorter_fr+` -- it's ever so slightly larger than
                    // `fr`.
                    let shorter_fr_plus = self.universal_regions.non_local_upper_bound(shorter_fr);
                    debug!(
                        "check_universal_region: shorter_fr_plus={:?}",
                        shorter_fr_plus
                    );

                    // Push the constraint `fr-: shorter_fr+`
                    propagated_outlives_requirements.push(ClosureOutlivesRequirement {
                        subject: ClosureOutlivesSubject::Region(fr_minus),
                        outlived_free_region: shorter_fr_plus,
                        blame_span: blame_span,
                    });
                    return;
                }
            }

            // If we are not in a context where we can propagate
            // errors, or we could not shrink `fr` to something
            // smaller, then just report an error.
            //
            // Note: in this case, we use the unapproximated regions
            // to report the error. This gives better error messages
            // in some cases.
            self.report_error(infcx, mir_def_id, longer_fr, shorter_fr, blame_span);
        }
    }

    /// Report an error because the universal region `fr` was required to outlive
    /// `outlived_fr` but it is not known to do so. For example:
    ///
    /// ```
    /// fn foo<'a, 'b>(x: &'a u32) -> &'b u32 { x }
    /// ```
    ///
    /// Here we would be invoked with `fr = 'a` and `outlived_fr = `'b`.
    fn report_error(
        &self,
        infcx: &InferCtxt<'_, '_, 'tcx>,
        mir_def_id: DefId,
        fr: RegionVid,
        outlived_fr: RegionVid,
        blame_span: Span,
    ) {
        // Obviously uncool error reporting.

        let fr_name = self.to_error_region(fr);
        let outlived_fr_name = self.to_error_region(outlived_fr);

        if let (Some(f), Some(o)) = (fr_name, outlived_fr_name) {
            let tables = infcx.tcx.typeck_tables_of(mir_def_id);
            let nice = NiceRegionError::new_from_span(infcx.tcx, blame_span, o, f, Some(tables));
            if let Some(ErrorReported) = nice.try_report() {
                return;
            }
        }

        let fr_string = match fr_name {
            Some(r) => format!("free region `{}`", r),
            None => format!("free region `{:?}`", fr),
        };

        let outlived_fr_string = match outlived_fr_name {
            Some(r) => format!("free region `{}`", r),
            None => format!("free region `{:?}`", outlived_fr),
        };

        let mut diag = infcx.tcx.sess.struct_span_err(
            blame_span,
            &format!("{} does not outlive {}", fr_string, outlived_fr_string,),
        );

        diag.emit();
    }

    crate fn why_region_contains_point(&self, fr1: RegionVid, elem: Location) -> Option<Cause> {
        // Find some constraint `X: Y` where:
        // - `fr1: X` transitively
        // - and `Y` is live at `elem`
        let index = self.blame_constraint(fr1, elem);
        let region_sub = self.constraints[index].sub;

        // then return why `Y` was live at `elem`
        self.liveness_constraints.cause(region_sub, elem)
    }

    /// Tries to finds a good span to blame for the fact that `fr1`
    /// contains `fr2`.
    fn blame_constraint(&self, fr1: RegionVid, elem: impl ToElementIndex) -> ConstraintIndex {
        // Find everything that influenced final value of `fr`.
        let influenced_fr1 = self.dependencies(fr1);

        // Try to find some outlives constraint `'X: fr2` where `'X`
        // influenced `fr1`. Blame that.
        //
        // NB, this is a pretty bad choice most of the time. In
        // particular, the connection between `'X` and `fr1` may not
        // be obvious to the user -- not to mention the naive notion
        // of dependencies, which doesn't account for the locations of
        // contraints at all. But it will do for now.
        let relevant_constraint = self.constraints
            .iter_enumerated()
            .filter_map(|(i, constraint)| {
                if !self.liveness_constraints.contains(constraint.sub, elem) {
                    None
                } else {
                    influenced_fr1[constraint.sup]
                        .map(|distance| (distance, i))
                }
            })
            .min() // constraining fr1 with fewer hops *ought* to be more obvious
            .map(|(_dist, i)| i);

        relevant_constraint.unwrap_or_else(|| {
            bug!(
                "could not find any constraint to blame for {:?}: {:?}",
                fr1,
                elem,
            );
        })
    }

    /// Finds all regions whose values `'a` may depend on in some way.
    /// For each region, returns either `None` (does not influence
    /// `'a`) or `Some(d)` which indicates that it influences `'a`
    /// with distinct `d` (minimum number of edges that must be
    /// traversed).
    ///
    /// Used during error reporting, extremely naive and inefficient.
    fn dependencies(&self, r0: RegionVid) -> IndexVec<RegionVid, Option<usize>> {
        let mut result_set = IndexVec::from_elem(None, &self.definitions);
        let mut changed = true;
        result_set[r0] = Some(0); // distance 0 from `r0`

        while changed {
            changed = false;
            for constraint in &self.constraints {
                if let Some(n) = result_set[constraint.sup] {
                    let m = n + 1;
                    if result_set[constraint.sub]
                        .map(|distance| m < distance)
                        .unwrap_or(true)
                    {
                        result_set[constraint.sub] = Some(m);
                        changed = true;
                    }
                }
            }
        }

        result_set
    }
}

impl<'tcx> RegionDefinition<'tcx> {
    fn new(origin: RegionVariableOrigin) -> Self {
        // Create a new region definition. Note that, for free
        // regions, these fields get updated later in
        // `init_universal_regions`.
        Self {
            origin,
            is_universal: false,
            external_name: None,
        }
    }
}

impl fmt::Debug for OutlivesConstraint {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "({:?}: {:?} @ {:?}) due to {:?}",
            self.sup, self.sub, self.point, self.span
        )
    }
}

pub trait ClosureRegionRequirementsExt<'gcx, 'tcx> {
    fn apply_requirements(
        &self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
        body_id: ast::NodeId,
        location: Location,
        closure_def_id: DefId,
        closure_substs: ty::ClosureSubsts<'tcx>,
    );

    fn subst_closure_mapping<T>(
        &self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
        closure_mapping: &IndexVec<RegionVid, ty::Region<'tcx>>,
        value: &T,
    ) -> T
    where
        T: TypeFoldable<'tcx>;
}

impl<'gcx, 'tcx> ClosureRegionRequirementsExt<'gcx, 'tcx> for ClosureRegionRequirements<'gcx> {
    /// Given an instance T of the closure type, this method
    /// instantiates the "extra" requirements that we computed for the
    /// closure into the inference context. This has the effect of
    /// adding new outlives obligations to existing variables.
    ///
    /// As described on `ClosureRegionRequirements`, the extra
    /// requirements are expressed in terms of regionvids that index
    /// into the free regions that appear on the closure type. So, to
    /// do this, we first copy those regions out from the type T into
    /// a vector. Then we can just index into that vector to extract
    /// out the corresponding region from T and apply the
    /// requirements.
    fn apply_requirements(
        &self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
        body_id: ast::NodeId,
        location: Location,
        closure_def_id: DefId,
        closure_substs: ty::ClosureSubsts<'tcx>,
    ) {
        let tcx = infcx.tcx;

        debug!(
            "apply_requirements(location={:?}, closure_def_id={:?}, closure_substs={:?})",
            location, closure_def_id, closure_substs
        );

        // Get Tu.
        let user_closure_ty = tcx.mk_closure(closure_def_id, closure_substs);
        debug!("apply_requirements: user_closure_ty={:?}", user_closure_ty);

        // Extract the values of the free regions in `user_closure_ty`
        // into a vector.  These are the regions that we will be
        // relating to one another.
        let closure_mapping =
            &UniversalRegions::closure_mapping(infcx, user_closure_ty, self.num_external_vids);
        debug!("apply_requirements: closure_mapping={:?}", closure_mapping);

        // Create the predicates.
        for outlives_requirement in &self.outlives_requirements {
            let outlived_region = closure_mapping[outlives_requirement.outlived_free_region];

            // FIXME, this origin is not entirely suitable.
            let origin = SubregionOrigin::CallRcvr(outlives_requirement.blame_span);

            match outlives_requirement.subject {
                ClosureOutlivesSubject::Region(region) => {
                    let region = closure_mapping[region];
                    debug!(
                        "apply_requirements: region={:?} \
                         outlived_region={:?} \
                         outlives_requirement={:?}",
                        region, outlived_region, outlives_requirement,
                    );
                    infcx.sub_regions(origin, outlived_region, region);
                }

                ClosureOutlivesSubject::Ty(ty) => {
                    let ty = self.subst_closure_mapping(infcx, closure_mapping, &ty);
                    debug!(
                        "apply_requirements: ty={:?} \
                         outlived_region={:?} \
                         outlives_requirement={:?}",
                        ty, outlived_region, outlives_requirement,
                    );
                    infcx.register_region_obligation(
                        body_id,
                        RegionObligation {
                            sup_type: ty,
                            sub_region: outlived_region,
                            cause: ObligationCause::misc(outlives_requirement.blame_span, body_id),
                        },
                    );
                }
            }
        }
    }

    fn subst_closure_mapping<T>(
        &self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
        closure_mapping: &IndexVec<RegionVid, ty::Region<'tcx>>,
        value: &T,
    ) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        infcx.tcx.fold_regions(value, &mut false, |r, _depth| {
            if let ty::ReClosureBound(vid) = r {
                closure_mapping[*vid]
            } else {
                bug!(
                    "subst_closure_mapping: encountered non-closure bound free region {:?}",
                    r
                )
            }
        })
    }
}
