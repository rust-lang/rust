use rustc_hir::def_id::LocalDefId;
use rustc_infer::infer::canonical::QueryRegionConstraints;
use rustc_infer::infer::outlives::env::RegionBoundPairs;
use rustc_infer::infer::outlives::obligations::{TypeOutlives, TypeOutlivesDelegate};
use rustc_infer::infer::region_constraints::{GenericKind, VerifyBound};
use rustc_infer::infer::{self, InferCtxt, SubregionOrigin};
use rustc_infer::traits::query::type_op::DeeplyNormalize;
use rustc_middle::bug;
use rustc_middle::ty::{
    self, GenericArgKind, Ty, TyCtxt, TypeFoldable, TypeVisitableExt, fold_regions,
};
use rustc_span::Span;
use rustc_trait_selection::traits::query::type_op::{TypeOp, TypeOpOutput};
use tracing::{debug, instrument};

use crate::constraints::OutlivesConstraint;
use crate::region_infer::TypeTest;
use crate::type_check::{Locations, MirTypeckRegionConstraints};
use crate::universal_regions::UniversalRegions;
use crate::{ClosureOutlivesSubject, ClosureRegionRequirements, ConstraintCategory};

pub(crate) struct ConstraintConversion<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
    tcx: TyCtxt<'tcx>,
    universal_regions: &'a UniversalRegions<'tcx>,
    /// Each RBP `GK: 'a` is assumed to be true. These encode
    /// relationships like `T: 'a` that are added via implicit bounds
    /// or the `param_env`.
    ///
    /// Each region here is guaranteed to be a key in the `indices`
    /// map. We use the "original" regions (i.e., the keys from the
    /// map, and not the values) because the code in
    /// `process_registered_region_obligations` has some special-cased
    /// logic expecting to see (e.g.) `ReStatic`, and if we supplied
    /// our special inference variable there, we would mess that up.
    region_bound_pairs: &'a RegionBoundPairs<'tcx>,
    implicit_region_bound: ty::Region<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    known_type_outlives_obligations: &'a [ty::PolyTypeOutlivesPredicate<'tcx>],
    locations: Locations,
    span: Span,
    category: ConstraintCategory<'tcx>,
    from_closure: bool,
    constraints: &'a mut MirTypeckRegionConstraints<'tcx>,
}

impl<'a, 'tcx> ConstraintConversion<'a, 'tcx> {
    pub(crate) fn new(
        infcx: &'a InferCtxt<'tcx>,
        universal_regions: &'a UniversalRegions<'tcx>,
        region_bound_pairs: &'a RegionBoundPairs<'tcx>,
        implicit_region_bound: ty::Region<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        known_type_outlives_obligations: &'a [ty::PolyTypeOutlivesPredicate<'tcx>],
        locations: Locations,
        span: Span,
        category: ConstraintCategory<'tcx>,
        constraints: &'a mut MirTypeckRegionConstraints<'tcx>,
    ) -> Self {
        Self {
            infcx,
            tcx: infcx.tcx,
            universal_regions,
            region_bound_pairs,
            implicit_region_bound,
            param_env,
            known_type_outlives_obligations,
            locations,
            span,
            category,
            constraints,
            from_closure: false,
        }
    }

    #[instrument(skip(self), level = "debug")]
    pub(super) fn convert_all(&mut self, query_constraints: &QueryRegionConstraints<'tcx>) {
        let QueryRegionConstraints { outlives } = query_constraints;

        for &(predicate, constraint_category) in outlives {
            self.convert(predicate, constraint_category);
        }
    }

    /// Given an instance of the closure type, this method instantiates the "extra" requirements
    /// that we computed for the closure. This has the effect of adding new outlives obligations
    /// to existing region variables in `closure_args`.
    #[instrument(skip(self), level = "debug")]
    pub(crate) fn apply_closure_requirements(
        &mut self,
        closure_requirements: &ClosureRegionRequirements<'tcx>,
        closure_def_id: LocalDefId,
        closure_args: ty::GenericArgsRef<'tcx>,
    ) {
        // Extract the values of the free regions in `closure_args`
        // into a vector. These are the regions that we will be
        // relating to one another.
        let closure_mapping = &UniversalRegions::closure_mapping(
            self.tcx,
            closure_args,
            closure_requirements.num_external_vids,
            closure_def_id,
        );
        debug!(?closure_mapping);

        // Create the predicates.
        let backup = (self.category, self.span, self.from_closure);
        self.from_closure = true;
        for outlives_requirement in &closure_requirements.outlives_requirements {
            let outlived_region = closure_mapping[outlives_requirement.outlived_free_region];
            let subject = match outlives_requirement.subject {
                ClosureOutlivesSubject::Region(re) => closure_mapping[re].into(),
                ClosureOutlivesSubject::Ty(subject_ty) => {
                    subject_ty.instantiate(self.tcx, |vid| closure_mapping[vid]).into()
                }
            };

            self.category = outlives_requirement.category;
            self.span = outlives_requirement.blame_span;
            self.convert(ty::OutlivesPredicate(subject, outlived_region), self.category);
        }
        (self.category, self.span, self.from_closure) = backup;
    }

    fn convert(
        &mut self,
        predicate: ty::OutlivesPredicate<'tcx, ty::GenericArg<'tcx>>,
        constraint_category: ConstraintCategory<'tcx>,
    ) {
        debug!("generate: constraints at: {:#?}", self.locations);

        // Extract out various useful fields we'll need below.
        let ConstraintConversion {
            tcx,
            infcx,
            region_bound_pairs,
            implicit_region_bound,
            known_type_outlives_obligations,
            ..
        } = *self;

        let mut outlives_predicates = vec![(predicate, constraint_category)];
        for iteration in 0.. {
            if outlives_predicates.is_empty() {
                break;
            }

            if !self.tcx.recursion_limit().value_within_limit(iteration) {
                bug!(
                    "FIXME(-Znext-solver): Overflowed when processing region obligations: {outlives_predicates:#?}"
                );
            }

            let mut next_outlives_predicates = vec![];
            for (ty::OutlivesPredicate(k1, r2), constraint_category) in outlives_predicates {
                match k1.unpack() {
                    GenericArgKind::Lifetime(r1) => {
                        let r1_vid = self.to_region_vid(r1);
                        let r2_vid = self.to_region_vid(r2);
                        self.add_outlives(r1_vid, r2_vid, constraint_category);
                    }

                    GenericArgKind::Type(mut t1) => {
                        // Normalize the type we receive from a `TypeOutlives` obligation
                        // in the new trait solver.
                        if infcx.next_trait_solver() {
                            t1 = self.normalize_and_add_type_outlives_constraints(
                                t1,
                                &mut next_outlives_predicates,
                            );
                        }

                        // we don't actually use this for anything, but
                        // the `TypeOutlives` code needs an origin.
                        let origin = infer::RelateParamBound(self.span, t1, None);

                        TypeOutlives::new(
                            &mut *self,
                            tcx,
                            region_bound_pairs,
                            Some(implicit_region_bound),
                            known_type_outlives_obligations,
                        )
                        .type_must_outlive(
                            origin,
                            t1,
                            r2,
                            constraint_category,
                        );
                    }

                    GenericArgKind::Const(_) => unreachable!(),
                }
            }

            outlives_predicates = next_outlives_predicates;
        }
    }

    /// Placeholder regions need to be converted eagerly because it may
    /// create new region variables, which we must not do when verifying
    /// our region bounds.
    ///
    /// FIXME: This should get removed once higher ranked region obligations
    /// are dealt with during trait solving.
    fn replace_placeholders_with_nll<T: TypeFoldable<TyCtxt<'tcx>>>(&mut self, value: T) -> T {
        if value.has_placeholders() {
            fold_regions(self.tcx, value, |r, _| match r.kind() {
                ty::RePlaceholder(placeholder) => {
                    self.constraints.placeholder_region(self.infcx, placeholder)
                }
                _ => r,
            })
        } else {
            value
        }
    }

    fn verify_to_type_test(
        &mut self,
        generic_kind: GenericKind<'tcx>,
        region: ty::Region<'tcx>,
        verify_bound: VerifyBound<'tcx>,
    ) -> TypeTest<'tcx> {
        let lower_bound = self.to_region_vid(region);
        TypeTest { generic_kind, lower_bound, span: self.span, verify_bound }
    }

    fn to_region_vid(&mut self, r: ty::Region<'tcx>) -> ty::RegionVid {
        if let ty::RePlaceholder(placeholder) = r.kind() {
            self.constraints.placeholder_region(self.infcx, placeholder).as_var()
        } else {
            self.universal_regions.to_region_vid(r)
        }
    }

    fn add_outlives(
        &mut self,
        sup: ty::RegionVid,
        sub: ty::RegionVid,
        category: ConstraintCategory<'tcx>,
    ) {
        let category = match self.category {
            ConstraintCategory::Boring | ConstraintCategory::BoringNoLocation => category,
            _ => self.category,
        };
        self.constraints.outlives_constraints.push(OutlivesConstraint {
            locations: self.locations,
            category,
            span: self.span,
            sub,
            sup,
            variance_info: ty::VarianceDiagInfo::default(),
            from_closure: self.from_closure,
        });
    }

    fn add_type_test(&mut self, type_test: TypeTest<'tcx>) {
        debug!("add_type_test(type_test={:?})", type_test);
        self.constraints.type_tests.push(type_test);
    }

    fn normalize_and_add_type_outlives_constraints(
        &self,
        ty: Ty<'tcx>,
        next_outlives_predicates: &mut Vec<(
            ty::OutlivesPredicate<'tcx, ty::GenericArg<'tcx>>,
            ConstraintCategory<'tcx>,
        )>,
    ) -> Ty<'tcx> {
        match self.param_env.and(DeeplyNormalize { value: ty }).fully_perform(self.infcx, self.span)
        {
            Ok(TypeOpOutput { output: ty, constraints, .. }) => {
                if let Some(QueryRegionConstraints { outlives }) = constraints {
                    next_outlives_predicates.extend(outlives.iter().copied());
                }
                ty
            }
            Err(_) => ty,
        }
    }
}

impl<'a, 'b, 'tcx> TypeOutlivesDelegate<'tcx> for &'a mut ConstraintConversion<'b, 'tcx> {
    fn push_sub_region_constraint(
        &mut self,
        _origin: SubregionOrigin<'tcx>,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
        constraint_category: ConstraintCategory<'tcx>,
    ) {
        let b = self.to_region_vid(b);
        let a = self.to_region_vid(a);
        self.add_outlives(b, a, constraint_category);
    }

    fn push_verify(
        &mut self,
        _origin: SubregionOrigin<'tcx>,
        kind: GenericKind<'tcx>,
        a: ty::Region<'tcx>,
        bound: VerifyBound<'tcx>,
    ) {
        let kind = self.replace_placeholders_with_nll(kind);
        let bound = self.replace_placeholders_with_nll(bound);
        let type_test = self.verify_to_type_test(kind, a, bound);
        self.add_type_test(type_test);
    }
}
