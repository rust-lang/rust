use std::fmt;

use rustc_errors::{Diag, E0275, EmissionGuarantee, ErrorGuaranteed, struct_span_code_err};
use rustc_hir::def::Namespace;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_hir::limit::Limit;
use rustc_infer::traits::{Obligation, PredicateObligation};
use rustc_middle::ty::print::{FmtPrinter, Print};
use rustc_middle::ty::{self, TyCtxt, Upcast};
use rustc_span::Span;
use tracing::debug;

use crate::error_reporting::TypeErrCtxt;

pub enum OverflowCause<'tcx> {
    DeeplyNormalize(ty::AliasTerm<'tcx>),
    TraitSolver(ty::Predicate<'tcx>),
}

pub fn suggest_new_overflow_limit<'tcx, G: EmissionGuarantee>(
    tcx: TyCtxt<'tcx>,
    err: &mut Diag<'_, G>,
) {
    let suggested_limit = match tcx.recursion_limit() {
        Limit(0) => Limit(2),
        limit => limit * 2,
    };
    err.help(format!(
        "consider increasing the recursion limit by adding a \
         `#![recursion_limit = \"{}\"]` attribute to your crate (`{}`)",
        suggested_limit,
        tcx.crate_name(LOCAL_CRATE),
    ));
}

impl<'a, 'tcx> TypeErrCtxt<'a, 'tcx> {
    /// Reports that an overflow has occurred and halts compilation. We
    /// halt compilation unconditionally because it is important that
    /// overflows never be masked -- they basically represent computations
    /// whose result could not be truly determined and thus we can't say
    /// if the program type checks or not -- and they are unusual
    /// occurrences in any case.
    pub fn report_overflow_error(
        &self,
        cause: OverflowCause<'tcx>,
        span: Span,
        suggest_increasing_limit: bool,
        mutate: impl FnOnce(&mut Diag<'_>),
    ) -> ! {
        let mut err = self.build_overflow_error(cause, span, suggest_increasing_limit);
        mutate(&mut err);
        err.emit().raise_fatal();
    }

    pub fn build_overflow_error(
        &self,
        cause: OverflowCause<'tcx>,
        span: Span,
        suggest_increasing_limit: bool,
    ) -> Diag<'a> {
        fn with_short_path<'tcx, T>(tcx: TyCtxt<'tcx>, value: T) -> String
        where
            T: fmt::Display + Print<'tcx, FmtPrinter<'tcx, 'tcx>>,
        {
            let s = value.to_string();
            if s.len() > 50 {
                // We don't need to save the type to a file, we will be talking about this type already
                // in a separate note when we explain the obligation, so it will be available that way.
                let mut p: FmtPrinter<'_, '_> =
                    FmtPrinter::new_with_limit(tcx, Namespace::TypeNS, Limit(6));
                value.print(&mut p).unwrap();
                p.into_buffer()
            } else {
                s
            }
        }

        let mut err = match cause {
            OverflowCause::DeeplyNormalize(alias_term) => {
                let alias_term = self.resolve_vars_if_possible(alias_term);
                let kind = alias_term.kind(self.tcx).descr();
                let alias_str = with_short_path(self.tcx, alias_term);
                struct_span_code_err!(
                    self.dcx(),
                    span,
                    E0275,
                    "overflow normalizing the {kind} `{alias_str}`",
                )
            }
            OverflowCause::TraitSolver(predicate) => {
                let predicate = self.resolve_vars_if_possible(predicate);
                match predicate.kind().skip_binder() {
                    ty::PredicateKind::Subtype(ty::SubtypePredicate { a, b, a_is_expected: _ })
                    | ty::PredicateKind::Coerce(ty::CoercePredicate { a, b }) => {
                        struct_span_code_err!(
                            self.dcx(),
                            span,
                            E0275,
                            "overflow assigning `{a}` to `{b}`",
                        )
                    }
                    _ => {
                        let pred_str = with_short_path(self.tcx, predicate);
                        struct_span_code_err!(
                            self.dcx(),
                            span,
                            E0275,
                            "overflow evaluating the requirement `{pred_str}`",
                        )
                    }
                }
            }
        };

        if suggest_increasing_limit {
            suggest_new_overflow_limit(self.tcx, &mut err);
        }

        err
    }

    /// Reports that an overflow has occurred and halts compilation. We
    /// halt compilation unconditionally because it is important that
    /// overflows never be masked -- they basically represent computations
    /// whose result could not be truly determined and thus we can't say
    /// if the program type checks or not -- and they are unusual
    /// occurrences in any case.
    pub fn report_overflow_obligation<T>(
        &self,
        obligation: &Obligation<'tcx, T>,
        suggest_increasing_limit: bool,
    ) -> !
    where
        T: Upcast<TyCtxt<'tcx>, ty::Predicate<'tcx>> + Clone,
    {
        let predicate = obligation.predicate.clone().upcast(self.tcx);
        let predicate = self.resolve_vars_if_possible(predicate);
        self.report_overflow_error(
            OverflowCause::TraitSolver(predicate),
            obligation.cause.span,
            suggest_increasing_limit,
            |err| {
                self.note_obligation_cause_code(
                    obligation.cause.body_id,
                    err,
                    predicate,
                    obligation.param_env,
                    obligation.cause.code(),
                    &mut vec![],
                    &mut Default::default(),
                );
            },
        );
    }

    /// Reports that a cycle was detected which led to overflow and halts
    /// compilation. This is equivalent to `report_overflow_obligation` except
    /// that we can give a more helpful error message (and, in particular,
    /// we do not suggest increasing the overflow limit, which is not
    /// going to help).
    pub fn report_overflow_obligation_cycle(&self, cycle: &[PredicateObligation<'tcx>]) -> ! {
        let cycle = self.resolve_vars_if_possible(cycle.to_owned());
        assert!(!cycle.is_empty());

        debug!(?cycle, "report_overflow_error_cycle");

        // The 'deepest' obligation is most likely to have a useful
        // cause 'backtrace'
        self.report_overflow_obligation(
            cycle.iter().max_by_key(|p| p.recursion_depth).unwrap(),
            false,
        );
    }

    pub fn report_overflow_no_abort(
        &self,
        obligation: PredicateObligation<'tcx>,
        suggest_increasing_limit: bool,
    ) -> ErrorGuaranteed {
        let obligation = self.resolve_vars_if_possible(obligation);
        let mut err = self.build_overflow_error(
            OverflowCause::TraitSolver(obligation.predicate),
            obligation.cause.span,
            suggest_increasing_limit,
        );
        self.note_obligation_cause(&mut err, &obligation);
        err.emit()
    }
}
