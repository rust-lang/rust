use crate::errors::{
    ActualImplExpectedKind, ActualImplExpectedLifetimeKind, ActualImplExplNotes,
    TraitPlaceholderMismatch, TyOrSig,
};
use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use crate::infer::ValuePairs;
use crate::infer::{SubregionOrigin, TypeTrace};
use crate::traits::{ObligationCause, ObligationCauseCode};
use rustc_data_structures::intern::Interned;
use rustc_errors::{DiagnosticBuilder, ErrorGuaranteed, IntoDiagnosticArg};
use rustc_hir::def::Namespace;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::error::ExpectedFound;
use rustc_middle::ty::print::{FmtPrinter, Print, RegionHighlightMode};
use rustc_middle::ty::GenericArgsRef;
use rustc_middle::ty::{self, RePlaceholder, Region, TyCtxt};

use std::fmt;

// HACK(eddyb) maybe move this in a more central location.
#[derive(Copy, Clone)]
pub struct Highlighted<'tcx, T> {
    tcx: TyCtxt<'tcx>,
    highlight: RegionHighlightMode<'tcx>,
    value: T,
}

impl<'tcx, T> IntoDiagnosticArg for Highlighted<'tcx, T>
where
    T: for<'a> Print<'tcx, FmtPrinter<'a, 'tcx>, Error = fmt::Error, Output = FmtPrinter<'a, 'tcx>>,
{
    fn into_diagnostic_arg(self) -> rustc_errors::DiagnosticArgValue<'static> {
        rustc_errors::DiagnosticArgValue::Str(self.to_string().into())
    }
}

impl<'tcx, T> Highlighted<'tcx, T> {
    fn map<U>(self, f: impl FnOnce(T) -> U) -> Highlighted<'tcx, U> {
        Highlighted { tcx: self.tcx, highlight: self.highlight, value: f(self.value) }
    }
}

impl<'tcx, T> fmt::Display for Highlighted<'tcx, T>
where
    T: for<'a> Print<'tcx, FmtPrinter<'a, 'tcx>, Error = fmt::Error, Output = FmtPrinter<'a, 'tcx>>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut printer = ty::print::FmtPrinter::new(self.tcx, Namespace::TypeNS);
        printer.region_highlight_mode = self.highlight;

        let s = self.value.print(printer)?.into_buffer();
        f.write_str(&s)
    }
}

impl<'tcx> NiceRegionError<'_, 'tcx> {
    /// When given a `ConcreteFailure` for a function with arguments containing a named region and
    /// an anonymous region, emit a descriptive diagnostic error.
    pub(super) fn try_report_placeholder_conflict(
        &self,
    ) -> Option<DiagnosticBuilder<'tcx, ErrorGuaranteed>> {
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
                SubregionOrigin::Subtype(box TypeTrace { cause, values }),
                sub_placeholder @ Region(Interned(RePlaceholder(_), _)),
                _,
                sup_placeholder @ Region(Interned(RePlaceholder(_), _)),
                _,
            )) => self.try_report_trait_placeholder_mismatch(
                Some(ty::Region::new_var(self.tcx(), *vid)),
                cause,
                Some(*sub_placeholder),
                Some(*sup_placeholder),
                values,
            ),

            Some(RegionResolutionError::SubSupConflict(
                vid,
                _,
                SubregionOrigin::Subtype(box TypeTrace { cause, values }),
                sub_placeholder @ Region(Interned(RePlaceholder(_), _)),
                _,
                _,
                _,
            )) => self.try_report_trait_placeholder_mismatch(
                Some(ty::Region::new_var(self.tcx(), *vid)),
                cause,
                Some(*sub_placeholder),
                None,
                values,
            ),

            Some(RegionResolutionError::SubSupConflict(
                vid,
                _,
                SubregionOrigin::Subtype(box TypeTrace { cause, values }),
                _,
                _,
                sup_placeholder @ Region(Interned(RePlaceholder(_), _)),
                _,
            )) => self.try_report_trait_placeholder_mismatch(
                Some(ty::Region::new_var(self.tcx(), *vid)),
                cause,
                None,
                Some(*sup_placeholder),
                values,
            ),

            Some(RegionResolutionError::SubSupConflict(
                vid,
                _,
                _,
                _,
                SubregionOrigin::Subtype(box TypeTrace { cause, values }),
                sup_placeholder @ Region(Interned(RePlaceholder(_), _)),
                _,
            )) => self.try_report_trait_placeholder_mismatch(
                Some(ty::Region::new_var(self.tcx(), *vid)),
                cause,
                None,
                Some(*sup_placeholder),
                values,
            ),

            Some(RegionResolutionError::UpperBoundUniverseConflict(
                vid,
                _,
                _,
                SubregionOrigin::Subtype(box TypeTrace { cause, values }),
                sup_placeholder @ Region(Interned(RePlaceholder(_), _)),
            )) => self.try_report_trait_placeholder_mismatch(
                Some(ty::Region::new_var(self.tcx(), *vid)),
                cause,
                None,
                Some(*sup_placeholder),
                values,
            ),

            Some(RegionResolutionError::ConcreteFailure(
                SubregionOrigin::Subtype(box TypeTrace { cause, values }),
                sub_region @ Region(Interned(RePlaceholder(_), _)),
                sup_region @ Region(Interned(RePlaceholder(_), _)),
            )) => self.try_report_trait_placeholder_mismatch(
                None,
                cause,
                Some(*sub_region),
                Some(*sup_region),
                values,
            ),

            Some(RegionResolutionError::ConcreteFailure(
                SubregionOrigin::Subtype(box TypeTrace { cause, values }),
                sub_region @ Region(Interned(RePlaceholder(_), _)),
                sup_region,
            )) => self.try_report_trait_placeholder_mismatch(
                (!sup_region.has_name()).then_some(*sup_region),
                cause,
                Some(*sub_region),
                None,
                values,
            ),

            Some(RegionResolutionError::ConcreteFailure(
                SubregionOrigin::Subtype(box TypeTrace { cause, values }),
                sub_region,
                sup_region @ Region(Interned(RePlaceholder(_), _)),
            )) => self.try_report_trait_placeholder_mismatch(
                (!sub_region.has_name()).then_some(*sub_region),
                cause,
                None,
                Some(*sup_region),
                values,
            ),

            _ => None,
        }
    }

    fn try_report_trait_placeholder_mismatch(
        &self,
        vid: Option<Region<'tcx>>,
        cause: &ObligationCause<'tcx>,
        sub_placeholder: Option<Region<'tcx>>,
        sup_placeholder: Option<Region<'tcx>>,
        value_pairs: &ValuePairs<'tcx>,
    ) -> Option<DiagnosticBuilder<'tcx, ErrorGuaranteed>> {
        let (expected_args, found_args, trait_def_id) = match value_pairs {
            ValuePairs::TraitRefs(ExpectedFound { expected, found })
                if expected.def_id == found.def_id =>
            {
                (expected.args, found.args, expected.def_id)
            }
            ValuePairs::PolyTraitRefs(ExpectedFound { expected, found })
                if expected.def_id() == found.def_id() =>
            {
                // It's possible that the placeholders come from a binder
                // outside of this value pair. Use `no_bound_vars` as a
                // simple heuristic for that.
                (expected.no_bound_vars()?.args, found.no_bound_vars()?.args, expected.def_id())
            }
            _ => return None,
        };

        Some(self.report_trait_placeholder_mismatch(
            vid,
            cause,
            sub_placeholder,
            sup_placeholder,
            trait_def_id,
            expected_args,
            found_args,
        ))
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
    #[instrument(level = "debug", skip(self))]
    fn report_trait_placeholder_mismatch(
        &self,
        vid: Option<Region<'tcx>>,
        cause: &ObligationCause<'tcx>,
        sub_placeholder: Option<Region<'tcx>>,
        sup_placeholder: Option<Region<'tcx>>,
        trait_def_id: DefId,
        expected_args: GenericArgsRef<'tcx>,
        actual_args: GenericArgsRef<'tcx>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let span = cause.span();

        let (leading_ellipsis, satisfy_span, where_span, dup_span, def_id) =
            if let ObligationCauseCode::ItemObligation(def_id)
            | ObligationCauseCode::ExprItemObligation(def_id, ..) = *cause.code()
            {
                (
                    true,
                    Some(span),
                    Some(self.tcx().def_span(def_id)),
                    None,
                    self.tcx().def_path_str(def_id),
                )
            } else {
                (false, None, None, Some(span), String::new())
            };

        let expected_trait_ref = self.cx.resolve_vars_if_possible(ty::TraitRef::new(
            self.cx.tcx,
            trait_def_id,
            expected_args,
        ));
        let actual_trait_ref = self.cx.resolve_vars_if_possible(ty::TraitRef::new(
            self.cx.tcx,
            trait_def_id,
            actual_args,
        ));

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

        let actual_self_ty_has_vid =
            self.tcx().any_free_region_meets(&actual_trait_ref.self_ty(), |r| Some(r) == vid);

        let expected_self_ty_has_vid =
            self.tcx().any_free_region_meets(&expected_trait_ref.self_ty(), |r| Some(r) == vid);

        let any_self_ty_has_vid = actual_self_ty_has_vid || expected_self_ty_has_vid;

        debug!(
            ?actual_has_vid,
            ?expected_has_vid,
            ?has_sub,
            ?has_sup,
            ?actual_self_ty_has_vid,
            ?expected_self_ty_has_vid,
        );

        let actual_impl_expl_notes = self.explain_actual_impl_that_was_found(
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
            leading_ellipsis,
        );

        self.tcx().sess.create_err(TraitPlaceholderMismatch {
            span,
            satisfy_span,
            where_span,
            dup_span,
            def_id,
            trait_def_id: self.tcx().def_path_str(trait_def_id),
            actual_impl_expl_notes,
        })
    }

    /// Add notes with details about the expected and actual trait refs, with attention to cases
    /// when placeholder regions are involved: either the trait or the self type containing
    /// them needs to be mentioned the closest to the placeholders.
    /// This makes the error messages read better, however at the cost of some complexity
    /// due to the number of combinations we have to deal with.
    fn explain_actual_impl_that_was_found(
        &self,
        sub_placeholder: Option<Region<'tcx>>,
        sup_placeholder: Option<Region<'tcx>>,
        has_sub: Option<usize>,
        has_sup: Option<usize>,
        expected_trait_ref: ty::TraitRef<'tcx>,
        actual_trait_ref: ty::TraitRef<'tcx>,
        vid: Option<Region<'tcx>>,
        expected_has_vid: Option<usize>,
        actual_has_vid: Option<usize>,
        any_self_ty_has_vid: bool,
        leading_ellipsis: bool,
    ) -> Vec<ActualImplExplNotes<'tcx>> {
        // The weird thing here with the `maybe_highlighting_region` calls and the
        // the match inside is meant to be like this:
        //
        // - The match checks whether the given things (placeholders, etc) appear
        //   in the types are about to print
        // - Meanwhile, the `maybe_highlighting_region` calls set up
        //   highlights so that, if they do appear, we will replace
        //   them `'0` and whatever. (This replacement takes place
        //   inside the closure given to `maybe_highlighting_region`.)
        //
        // There is some duplication between the calls -- i.e., the
        // `maybe_highlighting_region` checks if (e.g.) `has_sub` is
        // None, an then we check again inside the closure, but this
        // setup sort of minimized the number of calls and so form.

        let highlight_trait_ref = |trait_ref| Highlighted {
            tcx: self.tcx(),
            highlight: RegionHighlightMode::new(self.tcx()),
            value: trait_ref,
        };

        let same_self_type = actual_trait_ref.self_ty() == expected_trait_ref.self_ty();

        let mut expected_trait_ref = highlight_trait_ref(expected_trait_ref);
        expected_trait_ref.highlight.maybe_highlighting_region(sub_placeholder, has_sub);
        expected_trait_ref.highlight.maybe_highlighting_region(sup_placeholder, has_sup);

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

        let (kind, ty_or_sig, trait_path) = if same_self_type {
            let mut self_ty = expected_trait_ref.map(|tr| tr.self_ty());
            self_ty.highlight.maybe_highlighting_region(vid, actual_has_vid);

            if self_ty.value.is_closure() && self.tcx().is_fn_trait(expected_trait_ref.value.def_id)
            {
                let closure_sig = self_ty.map(|closure| {
                    if let ty::Closure(_, args) = closure.kind() {
                        self.tcx().signature_unclosure(
                            args.as_closure().sig(),
                            rustc_hir::Unsafety::Normal,
                        )
                    } else {
                        bug!("type is not longer closure");
                    }
                });
                (
                    ActualImplExpectedKind::Signature,
                    TyOrSig::ClosureSig(closure_sig),
                    expected_trait_ref.map(|tr| tr.print_only_trait_path()),
                )
            } else {
                (
                    ActualImplExpectedKind::Other,
                    TyOrSig::Ty(self_ty),
                    expected_trait_ref.map(|tr| tr.print_only_trait_path()),
                )
            }
        } else if passive_voice {
            (
                ActualImplExpectedKind::Passive,
                TyOrSig::Ty(expected_trait_ref.map(|tr| tr.self_ty())),
                expected_trait_ref.map(|tr| tr.print_only_trait_path()),
            )
        } else {
            (
                ActualImplExpectedKind::Other,
                TyOrSig::Ty(expected_trait_ref.map(|tr| tr.self_ty())),
                expected_trait_ref.map(|tr| tr.print_only_trait_path()),
            )
        };

        let (lt_kind, lifetime_1, lifetime_2) = match (has_sub, has_sup) {
            (Some(n1), Some(n2)) => {
                (ActualImplExpectedLifetimeKind::Two, std::cmp::min(n1, n2), std::cmp::max(n1, n2))
            }
            (Some(n), _) | (_, Some(n)) => (ActualImplExpectedLifetimeKind::Any, n, 0),
            (None, None) => {
                if let Some(n) = expected_has_vid {
                    (ActualImplExpectedLifetimeKind::Some, n, 0)
                } else {
                    (ActualImplExpectedLifetimeKind::Nothing, 0, 0)
                }
            }
        };

        let note_1 = ActualImplExplNotes::new_expected(
            kind,
            lt_kind,
            leading_ellipsis,
            ty_or_sig,
            trait_path,
            lifetime_1,
            lifetime_2,
        );

        let mut actual_trait_ref = highlight_trait_ref(actual_trait_ref);
        actual_trait_ref.highlight.maybe_highlighting_region(vid, actual_has_vid);

        let passive_voice = match actual_has_vid {
            Some(_) => any_self_ty_has_vid,
            None => true,
        };

        let trait_path = actual_trait_ref.map(|tr| tr.print_only_trait_path());
        let ty = actual_trait_ref.map(|tr| tr.self_ty()).to_string();
        let has_lifetime = actual_has_vid.is_some();
        let lifetime = actual_has_vid.unwrap_or_default();

        let note_2 = if same_self_type {
            ActualImplExplNotes::ButActuallyImplementsTrait { trait_path, has_lifetime, lifetime }
        } else if passive_voice {
            ActualImplExplNotes::ButActuallyImplementedForTy {
                trait_path,
                ty,
                has_lifetime,
                lifetime,
            }
        } else {
            ActualImplExplNotes::ButActuallyTyImplements { trait_path, ty, has_lifetime, lifetime }
        };

        vec![note_1, note_2]
    }
}
