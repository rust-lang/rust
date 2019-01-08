use hir::def_id::DefId;
use infer::error_reporting::nice_region_error::NiceRegionError;
use infer::lexical_region_resolve::RegionResolutionError;
use infer::ValuePairs;
use infer::{SubregionOrigin, TypeTrace};
use traits::{ObligationCause, ObligationCauseCode};
use ty;
use ty::error::ExpectedFound;
use ty::subst::Substs;
use util::common::ErrorReported;
use util::ppaux::RegionHighlightMode;

impl NiceRegionError<'me, 'gcx, 'tcx> {
    /// When given a `ConcreteFailure` for a function with arguments containing a named region and
    /// an anonymous region, emit an descriptive diagnostic error.
    pub(super) fn try_report_placeholder_conflict(&self) -> Option<ErrorReported> {
        match &self.error {
            ///////////////////////////////////////////////////////////////////////////
            // NB. The ordering of cases in this match is very
            // sensitive, because we are often matching against
            // specific cases and then using an `_` to match all
            // others.

            ///////////////////////////////////////////////////////////////////////////
            // Check for errors from comparing trait failures -- first
            // with two placeholders, then with one.
            Some(RegionResolutionError::SubSupConflict(
                vid,
                _,
                SubregionOrigin::Subtype(TypeTrace {
                    cause,
                    values: ValuePairs::TraitRefs(ExpectedFound { expected, found }),
                }),
                sub_placeholder @ ty::RePlaceholder(_),
                _,
                sup_placeholder @ ty::RePlaceholder(_),
            ))
                if expected.def_id == found.def_id =>
            {
                Some(self.try_report_placeholders_trait(
                    Some(self.tcx.mk_region(ty::ReVar(*vid))),
                    cause,
                    Some(sub_placeholder),
                    Some(sup_placeholder),
                    expected.def_id,
                    expected.substs,
                    found.substs,
                ))
            }

            Some(RegionResolutionError::SubSupConflict(
                vid,
                _,
                SubregionOrigin::Subtype(TypeTrace {
                    cause,
                    values: ValuePairs::TraitRefs(ExpectedFound { expected, found }),
                }),
                sub_placeholder @ ty::RePlaceholder(_),
                _,
                _,
            ))
                if expected.def_id == found.def_id =>
            {
                Some(self.try_report_placeholders_trait(
                    Some(self.tcx.mk_region(ty::ReVar(*vid))),
                    cause,
                    Some(sub_placeholder),
                    None,
                    expected.def_id,
                    expected.substs,
                    found.substs,
                ))
            }

            Some(RegionResolutionError::SubSupConflict(
                vid,
                _,
                SubregionOrigin::Subtype(TypeTrace {
                    cause,
                    values: ValuePairs::TraitRefs(ExpectedFound { expected, found }),
                }),
                _,
                _,
                sup_placeholder @ ty::RePlaceholder(_),
            ))
                if expected.def_id == found.def_id =>
            {
                Some(self.try_report_placeholders_trait(
                    Some(self.tcx.mk_region(ty::ReVar(*vid))),
                    cause,
                    None,
                    Some(*sup_placeholder),
                    expected.def_id,
                    expected.substs,
                    found.substs,
                ))
            }

            Some(RegionResolutionError::ConcreteFailure(
                SubregionOrigin::Subtype(TypeTrace {
                    cause,
                    values: ValuePairs::TraitRefs(ExpectedFound { expected, found }),
                }),
                sub_region @ ty::RePlaceholder(_),
                sup_region @ ty::RePlaceholder(_),
            ))
                if expected.def_id == found.def_id =>
            {
                Some(self.try_report_placeholders_trait(
                    None,
                    cause,
                    Some(*sub_region),
                    Some(*sup_region),
                    expected.def_id,
                    expected.substs,
                    found.substs,
                ))
            }

            Some(RegionResolutionError::ConcreteFailure(
                SubregionOrigin::Subtype(TypeTrace {
                    cause,
                    values: ValuePairs::TraitRefs(ExpectedFound { expected, found }),
                }),
                sub_region @ ty::RePlaceholder(_),
                sup_region,
            ))
                if expected.def_id == found.def_id =>
            {
                Some(self.try_report_placeholders_trait(
                    Some(sup_region),
                    cause,
                    Some(*sub_region),
                    None,
                    expected.def_id,
                    expected.substs,
                    found.substs,
                ))
            }

            Some(RegionResolutionError::ConcreteFailure(
                SubregionOrigin::Subtype(TypeTrace {
                    cause,
                    values: ValuePairs::TraitRefs(ExpectedFound { expected, found }),
                }),
                sub_region,
                sup_region @ ty::RePlaceholder(_),
            ))
                if expected.def_id == found.def_id =>
            {
                Some(self.try_report_placeholders_trait(
                    Some(sub_region),
                    cause,
                    None,
                    Some(*sup_region),
                    expected.def_id,
                    expected.substs,
                    found.substs,
                ))
            }

            _ => None,
        }
    }

    // error[E0308]: implementation of `Foo` does not apply to enough lifetimes
    //   --> /home/nmatsakis/tmp/foo.rs:12:5
    //    |
    // 12 |     all::<&'static u32>();
    //    |     ^^^^^^^^^^^^^^^^^^^ lifetime mismatch
    //    |
    //    = note: Due to a where-clause on the function `all`,
    //    = note: `T` must implement `...` for any two lifetimes `'1` and `'2`.
    //    = note: However, the type `T` only implements `...` for some specific lifetime `'2`.
    fn try_report_placeholders_trait(
        &self,
        vid: Option<ty::Region<'tcx>>,
        cause: &ObligationCause<'tcx>,
        sub_placeholder: Option<ty::Region<'tcx>>,
        sup_placeholder: Option<ty::Region<'tcx>>,
        trait_def_id: DefId,
        expected_substs: &'tcx Substs<'tcx>,
        actual_substs: &'tcx Substs<'tcx>,
    ) -> ErrorReported {
        let mut err = self.tcx.sess.struct_span_err(
            cause.span(&self.tcx),
            &format!(
                "implementation of `{}` is not general enough",
                self.tcx.item_path_str(trait_def_id),
            ),
        );

        match cause.code {
            ObligationCauseCode::ItemObligation(def_id) => {
                err.note(&format!(
                    "Due to a where-clause on `{}`,",
                    self.tcx.item_path_str(def_id),
                ));
            }
            _ => (),
        }

        let expected_trait_ref = ty::TraitRef {
            def_id: trait_def_id,
            substs: expected_substs,
        };
        let actual_trait_ref = ty::TraitRef {
            def_id: trait_def_id,
            substs: actual_substs,
        };

        // Search the expected and actual trait references to see (a)
        // whether the sub/sup placeholders appear in them (sometimes
        // you have a trait ref like `T: Foo<fn(&u8)>`, where the
        // placeholder was created as part of an inner type) and (b)
        // whether the inference variable appears. In each case,
        // assign a counter value in each case if so.
        let mut counter = 0;
        let mut has_sub = None;
        let mut has_sup = None;
        let mut has_vid = None;

        self.tcx.for_each_free_region(&expected_trait_ref, |r| {
            if Some(r) == sub_placeholder && has_sub.is_none() {
                has_sub = Some(counter);
                counter += 1;
            } else if Some(r) == sup_placeholder && has_sup.is_none() {
                has_sup = Some(counter);
                counter += 1;
            }
        });

        self.tcx.for_each_free_region(&actual_trait_ref, |r| {
            if Some(r) == vid && has_vid.is_none() {
                has_vid = Some(counter);
                counter += 1;
            }
        });

        let self_ty_has_vid = self
            .tcx
            .any_free_region_meets(&actual_trait_ref.self_ty(), |r| Some(r) == vid);

        RegionHighlightMode::maybe_highlighting_region(sub_placeholder, has_sub, || {
            RegionHighlightMode::maybe_highlighting_region(sup_placeholder, has_sup, || {
                match (has_sub, has_sup) {
                    (Some(n1), Some(n2)) => {
                        err.note(&format!(
                            "`{}` must implement `{}` \
                             for any two lifetimes `'{}` and `'{}`",
                            expected_trait_ref.self_ty(),
                            expected_trait_ref,
                            std::cmp::min(n1, n2),
                            std::cmp::max(n1, n2),
                        ));
                    }
                    (Some(n), _) | (_, Some(n)) => {
                        err.note(&format!(
                            "`{}` must implement `{}` \
                             for any lifetime `'{}`",
                            expected_trait_ref.self_ty(),
                            expected_trait_ref,
                            n,
                        ));
                    }
                    (None, None) => {
                        err.note(&format!(
                            "`{}` must implement `{}`",
                            expected_trait_ref.self_ty(),
                            expected_trait_ref,
                        ));
                    }
                }
            })
        });

        RegionHighlightMode::maybe_highlighting_region(vid, has_vid, || match has_vid {
            Some(n) => {
                if self_ty_has_vid {
                    err.note(&format!(
                        "but `{}` only implements `{}` for the lifetime `'{}`",
                        actual_trait_ref.self_ty(),
                        actual_trait_ref,
                        n
                    ));
                } else {
                    err.note(&format!(
                        "but `{}` only implements `{}` for some lifetime `'{}`",
                        actual_trait_ref.self_ty(),
                        actual_trait_ref,
                        n
                    ));
                }
            }
            None => {
                err.note(&format!(
                    "but `{}` only implements `{}`",
                    actual_trait_ref.self_ty(),
                    actual_trait_ref,
                ));
            }
        });

        err.emit();
        ErrorReported
    }
}
