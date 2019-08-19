use errors::DiagnosticBuilder;
use crate::hir::def::Namespace;
use crate::hir::def_id::DefId;
use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use crate::infer::ValuePairs;
use crate::infer::{SubregionOrigin, TypeTrace};
use crate::traits::{ObligationCause, ObligationCauseCode};
use crate::ty::{self, TyCtxt};
use crate::ty::error::ExpectedFound;
use crate::ty::subst::SubstsRef;
use crate::ty::print::{Print, RegionHighlightMode, FmtPrinter};

use std::fmt::{self, Write};

impl NiceRegionError<'me, 'tcx> {
    /// When given a `ConcreteFailure` for a function with arguments containing a named region and
    /// an anonymous region, emit a descriptive diagnostic error.
    pub(super) fn try_report_placeholder_conflict(&self) -> Option<DiagnosticBuilder<'me>> {
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
            )) if expected.def_id == found.def_id => Some(self.try_report_placeholders_trait(
                Some(self.tcx().mk_region(ty::ReVar(*vid))),
                cause,
                Some(sub_placeholder),
                Some(sup_placeholder),
                expected.def_id,
                expected.substs,
                found.substs,
            )),

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
            )) if expected.def_id == found.def_id => Some(self.try_report_placeholders_trait(
                Some(self.tcx().mk_region(ty::ReVar(*vid))),
                cause,
                Some(sub_placeholder),
                None,
                expected.def_id,
                expected.substs,
                found.substs,
            )),

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
            )) if expected.def_id == found.def_id => Some(self.try_report_placeholders_trait(
                Some(self.tcx().mk_region(ty::ReVar(*vid))),
                cause,
                None,
                Some(*sup_placeholder),
                expected.def_id,
                expected.substs,
                found.substs,
            )),

            Some(RegionResolutionError::SubSupConflict(
                vid,
                _,
                _,
                _,
                SubregionOrigin::Subtype(TypeTrace {
                    cause,
                    values: ValuePairs::TraitRefs(ExpectedFound { expected, found }),
                }),
                sup_placeholder @ ty::RePlaceholder(_),
            )) if expected.def_id == found.def_id => Some(self.try_report_placeholders_trait(
                Some(self.tcx().mk_region(ty::ReVar(*vid))),
                cause,
                None,
                Some(*sup_placeholder),
                expected.def_id,
                expected.substs,
                found.substs,
            )),

            Some(RegionResolutionError::ConcreteFailure(
                SubregionOrigin::Subtype(TypeTrace {
                    cause,
                    values: ValuePairs::TraitRefs(ExpectedFound { expected, found }),
                }),
                sub_region @ ty::RePlaceholder(_),
                sup_region @ ty::RePlaceholder(_),
            )) if expected.def_id == found.def_id => Some(self.try_report_placeholders_trait(
                None,
                cause,
                Some(*sub_region),
                Some(*sup_region),
                expected.def_id,
                expected.substs,
                found.substs,
            )),

            Some(RegionResolutionError::ConcreteFailure(
                SubregionOrigin::Subtype(TypeTrace {
                    cause,
                    values: ValuePairs::TraitRefs(ExpectedFound { expected, found }),
                }),
                sub_region @ ty::RePlaceholder(_),
                sup_region,
            )) if expected.def_id == found.def_id => Some(self.try_report_placeholders_trait(
                Some(sup_region),
                cause,
                Some(*sub_region),
                None,
                expected.def_id,
                expected.substs,
                found.substs,
            )),

            Some(RegionResolutionError::ConcreteFailure(
                SubregionOrigin::Subtype(TypeTrace {
                    cause,
                    values: ValuePairs::TraitRefs(ExpectedFound { expected, found }),
                }),
                sub_region,
                sup_region @ ty::RePlaceholder(_),
            )) if expected.def_id == found.def_id => Some(self.try_report_placeholders_trait(
                Some(sub_region),
                cause,
                None,
                Some(*sup_region),
                expected.def_id,
                expected.substs,
                found.substs,
            )),

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
        expected_substs: SubstsRef<'tcx>,
        actual_substs: SubstsRef<'tcx>,
    ) -> DiagnosticBuilder<'me> {
        debug!(
            "try_report_placeholders_trait(\
             vid={:?}, \
             sub_placeholder={:?}, \
             sup_placeholder={:?}, \
             trait_def_id={:?}, \
             expected_substs={:?}, \
             actual_substs={:?})",
            vid, sub_placeholder, sup_placeholder, trait_def_id, expected_substs, actual_substs
        );

        let mut err = self.tcx().sess.struct_span_err(
            cause.span(self.tcx()),
            &format!(
                "implementation of `{}` is not general enough",
                self.tcx().def_path_str(trait_def_id),
            ),
        );

        match cause.code {
            ObligationCauseCode::ItemObligation(def_id) => {
                err.note(&format!(
                    "Due to a where-clause on `{}`,",
                    self.tcx().def_path_str(def_id),
                ));
            }
            _ => (),
        }

        let expected_trait_ref = self.infcx.resolve_vars_if_possible(&ty::TraitRef {
            def_id: trait_def_id,
            substs: expected_substs,
        });
        let actual_trait_ref = self.infcx.resolve_vars_if_possible(&ty::TraitRef {
            def_id: trait_def_id,
            substs: actual_substs,
        });

        // Search the expected and actual trait references to see (a)
        // whether the sub/sup placeholders appear in them (sometimes
        // you have a trait ref like `T: Foo<fn(&u8)>`, where the
        // placeholder was created as part of an inner type) and (b)
        // whether the inference variable appears. In each case,
        // assign a counter value in each case if so.
        let mut counter = 0;
        let mut has_sub = None;
        let mut has_sup = None;

        let mut actual_has_vid = None;
        let mut expected_has_vid = None;

        self.tcx().for_each_free_region(&expected_trait_ref, |r| {
            if Some(r) == sub_placeholder && has_sub.is_none() {
                has_sub = Some(counter);
                counter += 1;
            } else if Some(r) == sup_placeholder && has_sup.is_none() {
                has_sup = Some(counter);
                counter += 1;
            }

            if Some(r) == vid && expected_has_vid.is_none() {
                expected_has_vid = Some(counter);
                counter += 1;
            }
        });

        self.tcx().for_each_free_region(&actual_trait_ref, |r| {
            if Some(r) == vid && actual_has_vid.is_none() {
                actual_has_vid = Some(counter);
                counter += 1;
            }
        });

        let actual_self_ty_has_vid = self
            .tcx()
            .any_free_region_meets(&actual_trait_ref.self_ty(), |r| Some(r) == vid);

        let expected_self_ty_has_vid = self
            .tcx()
            .any_free_region_meets(&expected_trait_ref.self_ty(), |r| Some(r) == vid);

        let any_self_ty_has_vid = actual_self_ty_has_vid || expected_self_ty_has_vid;

        debug!(
            "try_report_placeholders_trait: actual_has_vid={:?}",
            actual_has_vid
        );
        debug!(
            "try_report_placeholders_trait: expected_has_vid={:?}",
            expected_has_vid
        );
        debug!("try_report_placeholders_trait: has_sub={:?}", has_sub);
        debug!("try_report_placeholders_trait: has_sup={:?}", has_sup);
        debug!(
            "try_report_placeholders_trait: actual_self_ty_has_vid={:?}",
            actual_self_ty_has_vid
        );
        debug!(
            "try_report_placeholders_trait: expected_self_ty_has_vid={:?}",
            expected_self_ty_has_vid
        );

        self.explain_actual_impl_that_was_found(
            &mut err,
            sub_placeholder,
            sup_placeholder,
            has_sub,
            has_sup,
            expected_trait_ref,
            actual_trait_ref,
            vid,
            expected_has_vid,
            actual_has_vid,
            any_self_ty_has_vid,
        );

        err
    }

    /// Add notes with details about the expected and actual trait refs, with attention to cases
    /// when placeholder regions are involved: either the trait or the self type containing
    /// them needs to be mentioned the closest to the placeholders.
    /// This makes the error messages read better, however at the cost of some complexity
    /// due to the number of combinations we have to deal with.
    fn explain_actual_impl_that_was_found(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        sub_placeholder: Option<ty::Region<'tcx>>,
        sup_placeholder: Option<ty::Region<'tcx>>,
        has_sub: Option<usize>,
        has_sup: Option<usize>,
        expected_trait_ref: ty::TraitRef<'tcx>,
        actual_trait_ref: ty::TraitRef<'tcx>,
        vid: Option<ty::Region<'tcx>>,
        expected_has_vid: Option<usize>,
        actual_has_vid: Option<usize>,
        any_self_ty_has_vid: bool,
    ) {
        // HACK(eddyb) maybe move this in a more central location.
        #[derive(Copy, Clone)]
        struct Highlighted<'tcx, T> {
            tcx: TyCtxt<'tcx>,
            highlight: RegionHighlightMode,
            value: T,
        }

        impl<'tcx, T> Highlighted<'tcx, T> {
            fn map<U>(self, f: impl FnOnce(T) -> U) -> Highlighted<'tcx, U> {
                Highlighted {
                    tcx: self.tcx,
                    highlight: self.highlight,
                    value: f(self.value),
                }
            }
        }

        impl<'tcx, T> fmt::Display for Highlighted<'tcx, T>
        where
            T: for<'a, 'b, 'c> Print<
                'tcx,
                FmtPrinter<'a, 'tcx, &'b mut fmt::Formatter<'c>>,
                Error = fmt::Error,
            >,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut printer = ty::print::FmtPrinter::new(self.tcx, f, Namespace::TypeNS);
                printer.region_highlight_mode = self.highlight;

                self.value.print(printer)?;
                Ok(())
            }
        }

        // The weird thing here with the `maybe_highlighting_region` calls and the
        // the match inside is meant to be like this:
        //
        // - The match checks whether the given things (placeholders, etc) appear
        //   in the types are about to print
        // - Meanwhile, the `maybe_highlighting_region` calls set up
        //   highlights so that, if they do appear, we will replace
        //   them `'0` and whatever.  (This replacement takes place
        //   inside the closure given to `maybe_highlighting_region`.)
        //
        // There is some duplication between the calls -- i.e., the
        // `maybe_highlighting_region` checks if (e.g.) `has_sub` is
        // None, an then we check again inside the closure, but this
        // setup sort of minimized the number of calls and so form.

        let highlight_trait_ref = |trait_ref| Highlighted {
            tcx: self.tcx(),
            highlight: RegionHighlightMode::default(),
            value: trait_ref,
        };

        let mut expected_trait_ref = highlight_trait_ref(expected_trait_ref);
        expected_trait_ref.highlight.maybe_highlighting_region(sub_placeholder, has_sub);
        expected_trait_ref.highlight.maybe_highlighting_region(sup_placeholder, has_sup);
        err.note(&{
            let passive_voice = match (has_sub, has_sup) {
                (Some(_), _) | (_, Some(_)) => any_self_ty_has_vid,
                (None, None) => {
                    expected_trait_ref.highlight.maybe_highlighting_region(vid, expected_has_vid);
                    match expected_has_vid {
                        Some(_) => true,
                        None => any_self_ty_has_vid,
                    }
                }
            };

            let mut note = if passive_voice {
                format!(
                    "`{}` would have to be implemented for the type `{}`",
                    expected_trait_ref,
                    expected_trait_ref.map(|tr| tr.self_ty()),
                )
            } else {
                format!(
                    "`{}` must implement `{}`",
                    expected_trait_ref.map(|tr| tr.self_ty()),
                    expected_trait_ref,
                )
            };

            match (has_sub, has_sup) {
                (Some(n1), Some(n2)) => {
                    let _ = write!(note,
                        ", for any two lifetimes `'{}` and `'{}`",
                        std::cmp::min(n1, n2),
                        std::cmp::max(n1, n2),
                    );
                }
                (Some(n), _) | (_, Some(n)) => {
                    let _ = write!(note,
                        ", for any lifetime `'{}`",
                        n,
                    );
                }
                (None, None) => if let Some(n) = expected_has_vid {
                    let _ = write!(note,
                        ", for some specific lifetime `'{}`",
                        n,
                    );
                },
            }

            note
        });

        let mut actual_trait_ref = highlight_trait_ref(actual_trait_ref);
        actual_trait_ref.highlight.maybe_highlighting_region(vid, actual_has_vid);
        err.note(&{
            let passive_voice = match actual_has_vid {
                Some(_) => any_self_ty_has_vid,
                None => true,
            };

            let mut note = if passive_voice {
                format!(
                    "but `{}` is actually implemented for the type `{}`",
                    actual_trait_ref,
                    actual_trait_ref.map(|tr| tr.self_ty()),
                )
            } else {
                format!(
                    "but `{}` actually implements `{}`",
                    actual_trait_ref.map(|tr| tr.self_ty()),
                    actual_trait_ref,
                )
            };

            if let Some(n) = actual_has_vid {
                let _ = write!(note, ", for some specific lifetime `'{}`", n);
            }

            note
        });
    }
}
