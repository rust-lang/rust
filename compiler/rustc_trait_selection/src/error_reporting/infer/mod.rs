//! Error Reporting Code for the inference engine
//!
//! Because of the way inference, and in particular region inference,
//! works, it often happens that errors are not detected until far after
//! the relevant line of code has been type-checked. Therefore, there is
//! an elaborate system to track why a particular constraint in the
//! inference graph arose so that we can explain to the user what gave
//! rise to a particular error.
//!
//! The system is based around a set of "origin" types. An "origin" is the
//! reason that a constraint or inference variable arose. There are
//! different "origin" enums for different kinds of constraints/variables
//! (e.g., `TypeOrigin`, `RegionVariableOrigin`). An origin always has
//! a span, but also more information so that we can generate a meaningful
//! error message.
//!
//! Having a catalog of all the different reasons an error can arise is
//! also useful for other reasons, like cross-referencing FAQs etc, though
//! we are not really taking advantage of this yet.
//!
//! # Region Inference
//!
//! Region inference is particularly tricky because it always succeeds "in
//! the moment" and simply registers a constraint. Then, at the end, we
//! can compute the full graph and report errors, so we need to be able to
//! store and later report what gave rise to the conflicting constraints.
//!
//! # Subtype Trace
//!
//! Determining whether `T1 <: T2` often involves a number of subtypes and
//! subconstraints along the way. A "TypeTrace" is an extended version
//! of an origin that traces the types and other values that were being
//! compared. It is not necessarily comprehensive (in fact, at the time of
//! this writing it only tracks the root values being compared) but I'd
//! like to extend it to include significant "waypoints". For example, if
//! you are comparing `(T1, T2) <: (T3, T4)`, and the problem is that `T2
//! <: T4` fails, I'd like the trace to include enough information to say
//! "in the 2nd element of the tuple". Similarly, failures when comparing
//! arguments or return types in fn types should be able to cite the
//! specific position, etc.
//!
//! # Reality vs plan
//!
//! Of course, there is still a LOT of code in typeck that has yet to be
//! ported to this system, and which relies on string concatenation at the
//! time of error detection.

use std::borrow::Cow;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::{cmp, fmt, iter};

use rustc_abi::ExternAbi;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_errors::{
    Applicability, Diag, DiagStyledString, IntoDiagArg, MultiSpan, StringPart, pluralize,
};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::Visitor;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{self as hir};
use rustc_macros::extension;
use rustc_middle::bug;
use rustc_middle::dep_graph::DepContext;
use rustc_middle::traits::PatternOriginExpr;
use rustc_middle::ty::error::{ExpectedFound, TypeError, TypeErrorToStringExt};
use rustc_middle::ty::print::{
    PrintError, PrintTraitRefExt as _, WrapBinderMode, with_forced_trimmed_paths,
};
use rustc_middle::ty::{
    self, List, ParamEnv, Region, Ty, TyCtxt, TypeFoldable, TypeSuperVisitable, TypeVisitable,
    TypeVisitableExt,
};
use rustc_span::def_id::LOCAL_CRATE;
use rustc_span::{BytePos, DesugaringKind, Pos, Span, sym};
use tracing::{debug, instrument};

use crate::error_reporting::TypeErrCtxt;
use crate::errors::{ObligationCauseFailureCode, TypeErrorAdditionalDiags};
use crate::infer;
use crate::infer::relate::{self, RelateResult, TypeRelation};
use crate::infer::{InferCtxt, InferCtxtExt as _, TypeTrace, ValuePairs};
use crate::solve::deeply_normalize_for_diagnostics;
use crate::traits::{
    IfExpressionCause, MatchExpressionArmCause, ObligationCause, ObligationCauseCode,
};

mod note_and_explain;
mod suggest;

pub mod need_type_info;
pub mod nice_region_error;
pub mod region;
pub mod sub_relations;

/// Makes a valid string literal from a string by escaping special characters (" and \),
/// unless they are already escaped.
fn escape_literal(s: &str) -> String {
    let mut escaped = String::with_capacity(s.len());
    let mut chrs = s.chars().peekable();
    while let Some(first) = chrs.next() {
        match (first, chrs.peek()) {
            ('\\', Some(&delim @ '"') | Some(&delim @ '\'')) => {
                escaped.push('\\');
                escaped.push(delim);
                chrs.next();
            }
            ('"' | '\'', _) => {
                escaped.push('\\');
                escaped.push(first)
            }
            (c, _) => escaped.push(c),
        };
    }
    escaped
}

impl<'a, 'tcx> TypeErrCtxt<'a, 'tcx> {
    // [Note-Type-error-reporting]
    // An invariant is that anytime the expected or actual type is Error (the special
    // error type, meaning that an error occurred when typechecking this expression),
    // this is a derived error. The error cascaded from another error (that was already
    // reported), so it's not useful to display it to the user.
    // The following methods implement this logic.
    // They check if either the actual or expected type is Error, and don't print the error
    // in this case. The typechecker should only ever report type errors involving mismatched
    // types using one of these methods, and should not call span_err directly for such
    // errors.
    pub fn type_error_struct_with_diag<M>(
        &self,
        sp: Span,
        mk_diag: M,
        actual_ty: Ty<'tcx>,
    ) -> Diag<'a>
    where
        M: FnOnce(String) -> Diag<'a>,
    {
        let actual_ty = self.resolve_vars_if_possible(actual_ty);
        debug!("type_error_struct_with_diag({:?}, {:?})", sp, actual_ty);

        let mut err = mk_diag(self.ty_to_string(actual_ty));

        // Don't report an error if actual type is `Error`.
        if actual_ty.references_error() {
            err.downgrade_to_delayed_bug();
        }

        err
    }

    pub fn report_mismatched_types(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        expected: Ty<'tcx>,
        actual: Ty<'tcx>,
        err: TypeError<'tcx>,
    ) -> Diag<'a> {
        self.report_and_explain_type_error(
            TypeTrace::types(cause, expected, actual),
            param_env,
            err,
        )
    }

    pub fn report_mismatched_consts(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        expected: ty::Const<'tcx>,
        actual: ty::Const<'tcx>,
        err: TypeError<'tcx>,
    ) -> Diag<'a> {
        self.report_and_explain_type_error(
            TypeTrace::consts(cause, expected, actual),
            param_env,
            err,
        )
    }

    pub fn get_impl_future_output_ty(&self, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
        let (def_id, args) = match *ty.kind() {
            ty::Alias(_, ty::AliasTy { def_id, args, .. })
                if matches!(self.tcx.def_kind(def_id), DefKind::OpaqueTy) =>
            {
                (def_id, args)
            }
            ty::Alias(_, ty::AliasTy { def_id, args, .. })
                if self.tcx.is_impl_trait_in_trait(def_id) =>
            {
                (def_id, args)
            }
            _ => return None,
        };

        let future_trait = self.tcx.require_lang_item(LangItem::Future, None);
        let item_def_id = self.tcx.associated_item_def_ids(future_trait)[0];

        self.tcx
            .explicit_item_self_bounds(def_id)
            .iter_instantiated_copied(self.tcx, args)
            .find_map(|(predicate, _)| {
                predicate
                    .kind()
                    .map_bound(|kind| match kind {
                        ty::ClauseKind::Projection(projection_predicate)
                            if projection_predicate.projection_term.def_id == item_def_id =>
                        {
                            projection_predicate.term.as_type()
                        }
                        _ => None,
                    })
                    .no_bound_vars()
                    .flatten()
            })
    }

    /// Adds a note if the types come from similarly named crates
    fn check_and_note_conflicting_crates(&self, err: &mut Diag<'_>, terr: TypeError<'tcx>) -> bool {
        // FIXME(estebank): unify with `report_similar_impl_candidates`. The message is similar,
        // even if the logic needed to detect the case is very different.
        use hir::def_id::CrateNum;
        use rustc_hir::definitions::DisambiguatedDefPathData;
        use ty::GenericArg;
        use ty::print::Printer;

        struct AbsolutePathPrinter<'tcx> {
            tcx: TyCtxt<'tcx>,
            segments: Vec<String>,
        }

        impl<'tcx> Printer<'tcx> for AbsolutePathPrinter<'tcx> {
            fn tcx<'a>(&'a self) -> TyCtxt<'tcx> {
                self.tcx
            }

            fn print_region(&mut self, _region: ty::Region<'_>) -> Result<(), PrintError> {
                Err(fmt::Error)
            }

            fn print_type(&mut self, _ty: Ty<'tcx>) -> Result<(), PrintError> {
                Err(fmt::Error)
            }

            fn print_dyn_existential(
                &mut self,
                _predicates: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
            ) -> Result<(), PrintError> {
                Err(fmt::Error)
            }

            fn print_const(&mut self, _ct: ty::Const<'tcx>) -> Result<(), PrintError> {
                Err(fmt::Error)
            }

            fn path_crate(&mut self, cnum: CrateNum) -> Result<(), PrintError> {
                self.segments = vec![self.tcx.crate_name(cnum).to_string()];
                Ok(())
            }
            fn path_qualified(
                &mut self,
                _self_ty: Ty<'tcx>,
                _trait_ref: Option<ty::TraitRef<'tcx>>,
            ) -> Result<(), PrintError> {
                Err(fmt::Error)
            }

            fn path_append_impl(
                &mut self,
                _print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
                _disambiguated_data: &DisambiguatedDefPathData,
                _self_ty: Ty<'tcx>,
                _trait_ref: Option<ty::TraitRef<'tcx>>,
            ) -> Result<(), PrintError> {
                Err(fmt::Error)
            }
            fn path_append(
                &mut self,
                print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
                disambiguated_data: &DisambiguatedDefPathData,
            ) -> Result<(), PrintError> {
                print_prefix(self)?;
                self.segments.push(disambiguated_data.to_string());
                Ok(())
            }
            fn path_generic_args(
                &mut self,
                print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
                _args: &[GenericArg<'tcx>],
            ) -> Result<(), PrintError> {
                print_prefix(self)
            }
        }

        let report_path_match = |err: &mut Diag<'_>, did1: DefId, did2: DefId, ty: &str| -> bool {
            // Only report definitions from different crates. If both definitions
            // are from a local module we could have false positives, e.g.
            // let _ = [{struct Foo; Foo}, {struct Foo; Foo}];
            if did1.krate != did2.krate {
                let abs_path = |def_id| {
                    let mut printer = AbsolutePathPrinter { tcx: self.tcx, segments: vec![] };
                    printer.print_def_path(def_id, &[]).map(|_| printer.segments)
                };

                // We compare strings because DefPath can be different
                // for imported and non-imported crates
                let expected_str = self.tcx.def_path_str(did1);
                let found_str = self.tcx.def_path_str(did2);
                let Ok(expected_abs) = abs_path(did1) else { return false };
                let Ok(found_abs) = abs_path(did2) else { return false };
                let same_path = || -> Result<_, PrintError> {
                    Ok(expected_str == found_str || expected_abs == found_abs)
                };
                // We want to use as unique a type path as possible. If both types are "locally
                // known" by the same name, we use the "absolute path" which uses the original
                // crate name instead.
                let (expected, found) = if expected_str == found_str {
                    (expected_abs.join("::"), found_abs.join("::"))
                } else {
                    (expected_str.clone(), found_str.clone())
                };
                if same_path().unwrap_or(false) {
                    // We've displayed "expected `a::b`, found `a::b`". We add context to
                    // differentiate the different cases where that might happen.
                    let expected_crate_name = self.tcx.crate_name(did1.krate);
                    let found_crate_name = self.tcx.crate_name(did2.krate);
                    let same_crate = expected_crate_name == found_crate_name;
                    let expected_sp = self.tcx.def_span(did1);
                    let found_sp = self.tcx.def_span(did2);

                    let both_direct_dependencies = if !did1.is_local()
                        && !did2.is_local()
                        && let Some(data1) = self.tcx.extern_crate(did1.krate)
                        && let Some(data2) = self.tcx.extern_crate(did2.krate)
                        && data1.dependency_of == LOCAL_CRATE
                        && data2.dependency_of == LOCAL_CRATE
                    {
                        // If both crates are directly depended on, we don't want to mention that
                        // in the final message, as it is redundant wording.
                        // We skip the case of semver trick, where one version of the local crate
                        // depends on another version of itself by checking that both crates at play
                        // are not the current one.
                        true
                    } else {
                        false
                    };

                    let mut span: MultiSpan = vec![expected_sp, found_sp].into();
                    span.push_span_label(
                        self.tcx.def_span(did1),
                        format!("this is the expected {ty} `{expected}`"),
                    );
                    span.push_span_label(
                        self.tcx.def_span(did2),
                        format!("this is the found {ty} `{found}`"),
                    );
                    for def_id in [did1, did2] {
                        let crate_name = self.tcx.crate_name(def_id.krate);
                        if !def_id.is_local()
                            && let Some(data) = self.tcx.extern_crate(def_id.krate)
                        {
                            let descr = if same_crate {
                                "one version of".to_string()
                            } else {
                                format!("one {ty} comes from")
                            };
                            let dependency = if both_direct_dependencies {
                                if let rustc_session::cstore::ExternCrateSource::Extern(def_id) =
                                    data.src
                                    && let Some(name) = self.tcx.opt_item_name(def_id)
                                {
                                    format!(", which is renamed locally to `{name}`")
                                } else {
                                    String::new()
                                }
                            } else if data.dependency_of == LOCAL_CRATE {
                                ", as a direct dependency of the current crate".to_string()
                            } else {
                                let dep = self.tcx.crate_name(data.dependency_of);
                                format!(", as a dependency of crate `{dep}`")
                            };
                            span.push_span_label(
                                data.span,
                                format!("{descr} crate `{crate_name}` used here{dependency}"),
                            );
                        }
                    }
                    let msg = if (did1.is_local() || did2.is_local()) && same_crate {
                        format!(
                            "the crate `{expected_crate_name}` is compiled multiple times, \
                             possibly with different configurations",
                        )
                    } else if same_crate {
                        format!(
                            "two different versions of crate `{expected_crate_name}` are being \
                             used; two types coming from two different versions of the same crate \
                             are different types even if they look the same",
                        )
                    } else {
                        format!(
                            "two types coming from two different crates are different types even \
                             if they look the same",
                        )
                    };
                    err.span_note(span, msg);
                    if same_crate {
                        err.help("you can use `cargo tree` to explore your dependency tree");
                    }
                    return true;
                }
            }
            false
        };
        match terr {
            TypeError::Sorts(ref exp_found) => {
                // if they are both "path types", there's a chance of ambiguity
                // due to different versions of the same crate
                if let (&ty::Adt(exp_adt, _), &ty::Adt(found_adt, _)) =
                    (exp_found.expected.kind(), exp_found.found.kind())
                {
                    return report_path_match(err, exp_adt.did(), found_adt.did(), "type");
                }
            }
            TypeError::Traits(ref exp_found) => {
                return report_path_match(err, exp_found.expected, exp_found.found, "trait");
            }
            _ => (), // FIXME(#22750) handle traits and stuff
        }
        false
    }

    fn note_error_origin(
        &self,
        err: &mut Diag<'_>,
        cause: &ObligationCause<'tcx>,
        exp_found: Option<ty::error::ExpectedFound<Ty<'tcx>>>,
        terr: TypeError<'tcx>,
        param_env: Option<ParamEnv<'tcx>>,
    ) {
        match *cause.code() {
            ObligationCauseCode::Pattern {
                origin_expr: Some(origin_expr),
                span: Some(span),
                root_ty,
            } => {
                let expected_ty = self.resolve_vars_if_possible(root_ty);
                if !matches!(
                    expected_ty.kind(),
                    ty::Infer(ty::InferTy::TyVar(_) | ty::InferTy::FreshTy(_))
                ) {
                    // don't show type `_`
                    if span.desugaring_kind() == Some(DesugaringKind::ForLoop)
                        && let ty::Adt(def, args) = expected_ty.kind()
                        && Some(def.did()) == self.tcx.get_diagnostic_item(sym::Option)
                    {
                        err.span_label(
                            span,
                            format!("this is an iterator with items of type `{}`", args.type_at(0)),
                        );
                    } else {
                        let expected_ty = self.tcx.short_string(expected_ty, err.long_ty_path());
                        err.span_label(span, format!("this expression has type `{expected_ty}`"));
                    }
                }
                if let Some(ty::error::ExpectedFound { found, .. }) = exp_found
                    && let Ok(mut peeled_snippet) =
                        self.tcx.sess.source_map().span_to_snippet(origin_expr.peeled_span)
                {
                    // Parentheses are needed for cases like as casts.
                    // We use the peeled_span for deref suggestions.
                    // It's also safe to use for box, since box only triggers if there
                    // wasn't a reference to begin with.
                    if origin_expr.peeled_prefix_suggestion_parentheses {
                        peeled_snippet = format!("({peeled_snippet})");
                    }

                    // Try giving a box suggestion first, as it is a special case of the
                    // deref suggestion.
                    if expected_ty.boxed_ty() == Some(found) {
                        err.span_suggestion_verbose(
                            span,
                            "consider dereferencing the boxed value",
                            format!("*{peeled_snippet}"),
                            Applicability::MachineApplicable,
                        );
                    } else if let Some(param_env) = param_env
                        && let Some(prefix) = self.should_deref_suggestion_on_mismatch(
                            param_env,
                            found,
                            expected_ty,
                            origin_expr,
                        )
                    {
                        err.span_suggestion_verbose(
                            span,
                            "consider dereferencing to access the inner value using the Deref trait",
                            format!("{prefix}{peeled_snippet}"),
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
            }
            ObligationCauseCode::Pattern { origin_expr: None, span: Some(span), .. } => {
                err.span_label(span, "expected due to this");
            }
            ObligationCauseCode::BlockTailExpression(
                _,
                hir::MatchSource::TryDesugar(scrut_hir_id),
            ) => {
                if let Some(ty::error::ExpectedFound { expected, .. }) = exp_found {
                    let scrut_expr = self.tcx.hir_expect_expr(scrut_hir_id);
                    let scrut_ty = if let hir::ExprKind::Call(_, args) = &scrut_expr.kind {
                        let arg_expr = args.first().expect("try desugaring call w/out arg");
                        self.typeck_results
                            .as_ref()
                            .and_then(|typeck_results| typeck_results.expr_ty_opt(arg_expr))
                    } else {
                        bug!("try desugaring w/out call expr as scrutinee");
                    };

                    match scrut_ty {
                        Some(ty) if expected == ty => {
                            let source_map = self.tcx.sess.source_map();
                            err.span_suggestion(
                                source_map.end_point(cause.span),
                                "try removing this `?`",
                                "",
                                Applicability::MachineApplicable,
                            );
                        }
                        _ => {}
                    }
                }
            }
            ObligationCauseCode::MatchExpressionArm(box MatchExpressionArmCause {
                arm_block_id,
                arm_span,
                arm_ty,
                prior_arm_block_id,
                prior_arm_span,
                prior_arm_ty,
                source,
                ref prior_non_diverging_arms,
                scrut_span,
                expr_span,
                ..
            }) => match source {
                hir::MatchSource::TryDesugar(scrut_hir_id) => {
                    if let Some(ty::error::ExpectedFound { expected, .. }) = exp_found {
                        let scrut_expr = self.tcx.hir_expect_expr(scrut_hir_id);
                        let scrut_ty = if let hir::ExprKind::Call(_, args) = &scrut_expr.kind {
                            let arg_expr = args.first().expect("try desugaring call w/out arg");
                            self.typeck_results
                                .as_ref()
                                .and_then(|typeck_results| typeck_results.expr_ty_opt(arg_expr))
                        } else {
                            bug!("try desugaring w/out call expr as scrutinee");
                        };

                        match scrut_ty {
                            Some(ty) if expected == ty => {
                                let source_map = self.tcx.sess.source_map();
                                err.span_suggestion(
                                    source_map.end_point(cause.span),
                                    "try removing this `?`",
                                    "",
                                    Applicability::MachineApplicable,
                                );
                            }
                            _ => {}
                        }
                    }
                }
                _ => {
                    // `prior_arm_ty` can be `!`, `expected` will have better info when present.
                    let t = self.resolve_vars_if_possible(match exp_found {
                        Some(ty::error::ExpectedFound { expected, .. }) => expected,
                        _ => prior_arm_ty,
                    });
                    let source_map = self.tcx.sess.source_map();
                    let mut any_multiline_arm = source_map.is_multiline(arm_span);
                    if prior_non_diverging_arms.len() <= 4 {
                        for sp in prior_non_diverging_arms {
                            any_multiline_arm |= source_map.is_multiline(*sp);
                            err.span_label(*sp, format!("this is found to be of type `{t}`"));
                        }
                    } else if let Some(sp) = prior_non_diverging_arms.last() {
                        any_multiline_arm |= source_map.is_multiline(*sp);
                        err.span_label(
                            *sp,
                            format!("this and all prior arms are found to be of type `{t}`"),
                        );
                    }
                    let outer = if any_multiline_arm || !source_map.is_multiline(expr_span) {
                        // Cover just `match` and the scrutinee expression, not
                        // the entire match body, to reduce diagram noise.
                        expr_span.shrink_to_lo().to(scrut_span)
                    } else {
                        expr_span
                    };
                    let msg = "`match` arms have incompatible types";
                    err.span_label(outer, msg);
                    if let Some(subdiag) = self.suggest_remove_semi_or_return_binding(
                        prior_arm_block_id,
                        prior_arm_ty,
                        prior_arm_span,
                        arm_block_id,
                        arm_ty,
                        arm_span,
                    ) {
                        err.subdiagnostic(subdiag);
                    }
                }
            },
            ObligationCauseCode::IfExpression(box IfExpressionCause {
                then_id,
                else_id,
                then_ty,
                else_ty,
                outer_span,
                ..
            }) => {
                let then_span = self.find_block_span_from_hir_id(then_id);
                let else_span = self.find_block_span_from_hir_id(else_id);
                if let hir::Node::Expr(e) = self.tcx.hir_node(else_id)
                    && let hir::ExprKind::If(_cond, _then, None) = e.kind
                    && else_ty.is_unit()
                {
                    // Account for `let x = if a { 1 } else if b { 2 };`
                    err.note("`if` expressions without `else` evaluate to `()`");
                    err.note("consider adding an `else` block that evaluates to the expected type");
                }
                err.span_label(then_span, "expected because of this");
                if let Some(sp) = outer_span {
                    err.span_label(sp, "`if` and `else` have incompatible types");
                }
                if let Some(subdiag) = self.suggest_remove_semi_or_return_binding(
                    Some(then_id),
                    then_ty,
                    then_span,
                    Some(else_id),
                    else_ty,
                    else_span,
                ) {
                    err.subdiagnostic(subdiag);
                }
            }
            ObligationCauseCode::LetElse => {
                err.help("try adding a diverging expression, such as `return` or `panic!(..)`");
                err.help("...or use `match` instead of `let...else`");
            }
            _ => {
                if let ObligationCauseCode::WhereClause(_, span)
                | ObligationCauseCode::WhereClauseInExpr(_, span, ..) =
                    cause.code().peel_derives()
                    && !span.is_dummy()
                    && let TypeError::RegionsPlaceholderMismatch = terr
                {
                    err.span_note(*span, "the lifetime requirement is introduced here");
                }
            }
        }
    }

    /// Determines whether deref_to == <deref_from as Deref>::Target, and if so,
    /// returns a prefix that should be added to deref_from as a suggestion.
    fn should_deref_suggestion_on_mismatch(
        &self,
        param_env: ParamEnv<'tcx>,
        deref_to: Ty<'tcx>,
        deref_from: Ty<'tcx>,
        origin_expr: PatternOriginExpr,
    ) -> Option<String> {
        // origin_expr contains stripped away versions of our expression.
        // We'll want to use that to avoid suggesting things like *&x.
        // However, the type that we have access to hasn't been stripped away,
        // so we need to ignore the first n dereferences, where n is the number
        // that's been stripped away in origin_expr.

        // Find a way to autoderef from deref_from to deref_to.
        let Some((num_derefs, (after_deref_ty, _))) = (self.autoderef_steps)(deref_from)
            .into_iter()
            .enumerate()
            .find(|(_, (ty, _))| self.infcx.can_eq(param_env, *ty, deref_to))
        else {
            return None;
        };

        if num_derefs <= origin_expr.peeled_count {
            return None;
        }

        let deref_part = "*".repeat(num_derefs - origin_expr.peeled_count);

        // If the user used a reference in the original expression, they probably
        // want the suggestion to still give a reference.
        if deref_from.is_ref() && !after_deref_ty.is_ref() {
            Some(format!("&{deref_part}"))
        } else {
            Some(deref_part)
        }
    }

    /// Given that `other_ty` is the same as a type argument for `name` in `sub`, populate `value`
    /// highlighting `name` and every type argument that isn't at `pos` (which is `other_ty`), and
    /// populate `other_value` with `other_ty`.
    ///
    /// ```text
    /// Foo<Bar<Qux>>
    /// ^^^^--------^ this is highlighted
    /// |   |
    /// |   this type argument is exactly the same as the other type, not highlighted
    /// this is highlighted
    /// Bar<Qux>
    /// -------- this type is the same as a type argument in the other type, not highlighted
    /// ```
    fn highlight_outer(
        &self,
        value: &mut DiagStyledString,
        other_value: &mut DiagStyledString,
        name: String,
        args: &[ty::GenericArg<'tcx>],
        pos: usize,
        other_ty: Ty<'tcx>,
    ) {
        // `value` and `other_value` hold two incomplete type representation for display.
        // `name` is the path of both types being compared. `sub`
        value.push_highlighted(name);

        if args.is_empty() {
            return;
        }
        value.push_highlighted("<");

        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                value.push_normal(", ");
            }

            match arg.kind() {
                ty::GenericArgKind::Lifetime(lt) => {
                    let s = lt.to_string();
                    value.push_normal(if s.is_empty() { "'_" } else { &s });
                }
                ty::GenericArgKind::Const(ct) => {
                    value.push_normal(ct.to_string());
                }
                // Highlight all the type arguments that aren't at `pos` and compare
                // the type argument at `pos` and `other_ty`.
                ty::GenericArgKind::Type(type_arg) => {
                    if i == pos {
                        let values = self.cmp(type_arg, other_ty);
                        value.0.extend((values.0).0);
                        other_value.0.extend((values.1).0);
                    } else {
                        value.push_highlighted(type_arg.to_string());
                    }
                }
            }
        }

        value.push_highlighted(">");
    }

    /// If `other_ty` is the same as a type argument present in `sub`, highlight `path` in `t1_out`,
    /// as that is the difference to the other type.
    ///
    /// For the following code:
    ///
    /// ```ignore (illustrative)
    /// let x: Foo<Bar<Qux>> = foo::<Bar<Qux>>();
    /// ```
    ///
    /// The type error output will behave in the following way:
    ///
    /// ```text
    /// Foo<Bar<Qux>>
    /// ^^^^--------^ this is highlighted
    /// |   |
    /// |   this type argument is exactly the same as the other type, not highlighted
    /// this is highlighted
    /// Bar<Qux>
    /// -------- this type is the same as a type argument in the other type, not highlighted
    /// ```
    fn cmp_type_arg(
        &self,
        t1_out: &mut DiagStyledString,
        t2_out: &mut DiagStyledString,
        path: String,
        args: &'tcx [ty::GenericArg<'tcx>],
        other_path: String,
        other_ty: Ty<'tcx>,
    ) -> bool {
        for (i, arg) in args.iter().enumerate() {
            if let Some(ta) = arg.as_type() {
                if ta == other_ty {
                    self.highlight_outer(t1_out, t2_out, path, args, i, other_ty);
                    return true;
                }
                if let ty::Adt(def, _) = ta.kind() {
                    let path_ = self.tcx.def_path_str(def.did());
                    if path_ == other_path {
                        self.highlight_outer(t1_out, t2_out, path, args, i, other_ty);
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Adds a `,` to the type representation only if it is appropriate.
    fn push_comma(
        &self,
        value: &mut DiagStyledString,
        other_value: &mut DiagStyledString,
        pos: usize,
    ) {
        if pos > 0 {
            value.push_normal(", ");
            other_value.push_normal(", ");
        }
    }

    /// Given two `fn` signatures highlight only sub-parts that are different.
    fn cmp_fn_sig(
        &self,
        sig1: &ty::PolyFnSig<'tcx>,
        fn_def1: Option<(DefId, Option<&'tcx [ty::GenericArg<'tcx>]>)>,
        sig2: &ty::PolyFnSig<'tcx>,
        fn_def2: Option<(DefId, Option<&'tcx [ty::GenericArg<'tcx>]>)>,
    ) -> (DiagStyledString, DiagStyledString) {
        let sig1 = &(self.normalize_fn_sig)(*sig1);
        let sig2 = &(self.normalize_fn_sig)(*sig2);

        let get_lifetimes = |sig| {
            use rustc_hir::def::Namespace;
            let (sig, reg) = ty::print::FmtPrinter::new(self.tcx, Namespace::TypeNS)
                .name_all_regions(sig, WrapBinderMode::ForAll)
                .unwrap();
            let lts: Vec<String> =
                reg.into_items().map(|(_, kind)| kind.to_string()).into_sorted_stable_ord();
            (if lts.is_empty() { String::new() } else { format!("for<{}> ", lts.join(", ")) }, sig)
        };

        let (lt1, sig1) = get_lifetimes(sig1);
        let (lt2, sig2) = get_lifetimes(sig2);

        // unsafe extern "C" for<'a> fn(&'a T) -> &'a T
        let mut values =
            (DiagStyledString::normal("".to_string()), DiagStyledString::normal("".to_string()));

        // unsafe extern "C" for<'a> fn(&'a T) -> &'a T
        // ^^^^^^
        let safety = |fn_def, sig: ty::FnSig<'_>| match fn_def {
            None => sig.safety.prefix_str(),
            Some((did, _)) => {
                if self.tcx.codegen_fn_attrs(did).safe_target_features {
                    "#[target_features] "
                } else {
                    sig.safety.prefix_str()
                }
            }
        };
        let safety1 = safety(fn_def1, sig1);
        let safety2 = safety(fn_def2, sig2);
        values.0.push(safety1, safety1 != safety2);
        values.1.push(safety2, safety1 != safety2);

        // unsafe extern "C" for<'a> fn(&'a T) -> &'a T
        //        ^^^^^^^^^^
        if sig1.abi != ExternAbi::Rust {
            values.0.push(format!("extern {} ", sig1.abi), sig1.abi != sig2.abi);
        }
        if sig2.abi != ExternAbi::Rust {
            values.1.push(format!("extern {} ", sig2.abi), sig1.abi != sig2.abi);
        }

        // unsafe extern "C" for<'a> fn(&'a T) -> &'a T
        //                   ^^^^^^^^
        let lifetime_diff = lt1 != lt2;
        values.0.push(lt1, lifetime_diff);
        values.1.push(lt2, lifetime_diff);

        // unsafe extern "C" for<'a> fn(&'a T) -> &'a T
        //                           ^^^
        values.0.push_normal("fn(");
        values.1.push_normal("fn(");

        // unsafe extern "C" for<'a> fn(&'a T) -> &'a T
        //                              ^^^^^
        let len1 = sig1.inputs().len();
        let len2 = sig2.inputs().len();
        if len1 == len2 {
            for (i, (l, r)) in iter::zip(sig1.inputs(), sig2.inputs()).enumerate() {
                self.push_comma(&mut values.0, &mut values.1, i);
                let (x1, x2) = self.cmp(*l, *r);
                (values.0).0.extend(x1.0);
                (values.1).0.extend(x2.0);
            }
        } else {
            for (i, l) in sig1.inputs().iter().enumerate() {
                values.0.push_highlighted(l.to_string());
                if i != len1 - 1 {
                    values.0.push_highlighted(", ");
                }
            }
            for (i, r) in sig2.inputs().iter().enumerate() {
                values.1.push_highlighted(r.to_string());
                if i != len2 - 1 {
                    values.1.push_highlighted(", ");
                }
            }
        }

        if sig1.c_variadic {
            if len1 > 0 {
                values.0.push_normal(", ");
            }
            values.0.push("...", !sig2.c_variadic);
        }
        if sig2.c_variadic {
            if len2 > 0 {
                values.1.push_normal(", ");
            }
            values.1.push("...", !sig1.c_variadic);
        }

        // unsafe extern "C" for<'a> fn(&'a T) -> &'a T
        //                                   ^
        values.0.push_normal(")");
        values.1.push_normal(")");

        // unsafe extern "C" for<'a> fn(&'a T) -> &'a T
        //                                     ^^^^^^^^
        let output1 = sig1.output();
        let output2 = sig2.output();
        let (x1, x2) = self.cmp(output1, output2);
        let output_diff = x1 != x2;
        if !output1.is_unit() || output_diff {
            values.0.push_normal(" -> ");
            (values.0).0.extend(x1.0);
        }
        if !output2.is_unit() || output_diff {
            values.1.push_normal(" -> ");
            (values.1).0.extend(x2.0);
        }

        let fmt = |did, args| format!(" {{{}}}", self.tcx.def_path_str_with_args(did, args));

        match (fn_def1, fn_def2) {
            (Some((fn_def1, Some(fn_args1))), Some((fn_def2, Some(fn_args2)))) => {
                let path1 = fmt(fn_def1, fn_args1);
                let path2 = fmt(fn_def2, fn_args2);
                let same_path = path1 == path2;
                values.0.push(path1, !same_path);
                values.1.push(path2, !same_path);
            }
            (Some((fn_def1, Some(fn_args1))), None) => {
                values.0.push_highlighted(fmt(fn_def1, fn_args1));
            }
            (None, Some((fn_def2, Some(fn_args2)))) => {
                values.1.push_highlighted(fmt(fn_def2, fn_args2));
            }
            _ => {}
        }

        values
    }

    pub fn cmp_traits(
        &self,
        def_id1: DefId,
        args1: &[ty::GenericArg<'tcx>],
        def_id2: DefId,
        args2: &[ty::GenericArg<'tcx>],
    ) -> (DiagStyledString, DiagStyledString) {
        let mut values = (DiagStyledString::new(), DiagStyledString::new());

        if def_id1 != def_id2 {
            values.0.push_highlighted(self.tcx.def_path_str(def_id1).as_str());
            values.1.push_highlighted(self.tcx.def_path_str(def_id2).as_str());
        } else {
            values.0.push_normal(self.tcx.item_name(def_id1).as_str());
            values.1.push_normal(self.tcx.item_name(def_id2).as_str());
        }

        if args1.len() != args2.len() {
            let (pre, post) = if args1.len() > 0 { ("<", ">") } else { ("", "") };
            values.0.push_normal(format!(
                "{pre}{}{post}",
                args1.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", ")
            ));
            let (pre, post) = if args2.len() > 0 { ("<", ">") } else { ("", "") };
            values.1.push_normal(format!(
                "{pre}{}{post}",
                args2.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", ")
            ));
            return values;
        }

        if args1.len() > 0 {
            values.0.push_normal("<");
            values.1.push_normal("<");
        }
        for (i, (a, b)) in std::iter::zip(args1, args2).enumerate() {
            let a_str = a.to_string();
            let b_str = b.to_string();
            if let (Some(a), Some(b)) = (a.as_type(), b.as_type()) {
                let (a, b) = self.cmp(a, b);
                values.0.0.extend(a.0);
                values.1.0.extend(b.0);
            } else if a_str != b_str {
                values.0.push_highlighted(a_str);
                values.1.push_highlighted(b_str);
            } else {
                values.0.push_normal(a_str);
                values.1.push_normal(b_str);
            }
            if i + 1 < args1.len() {
                values.0.push_normal(", ");
                values.1.push_normal(", ");
            }
        }
        if args1.len() > 0 {
            values.0.push_normal(">");
            values.1.push_normal(">");
        }
        values
    }

    /// Compares two given types, eliding parts that are the same between them and highlighting
    /// relevant differences, and return two representation of those types for highlighted printing.
    pub fn cmp(&self, t1: Ty<'tcx>, t2: Ty<'tcx>) -> (DiagStyledString, DiagStyledString) {
        debug!("cmp(t1={}, t1.kind={:?}, t2={}, t2.kind={:?})", t1, t1.kind(), t2, t2.kind());

        // helper functions
        let recurse = |t1, t2, values: &mut (DiagStyledString, DiagStyledString)| {
            let (x1, x2) = self.cmp(t1, t2);
            (values.0).0.extend(x1.0);
            (values.1).0.extend(x2.0);
        };

        fn fmt_region<'tcx>(region: ty::Region<'tcx>) -> String {
            let mut r = region.to_string();
            if r == "'_" {
                r.clear();
            } else {
                r.push(' ');
            }
            format!("&{r}")
        }

        fn push_ref<'tcx>(
            region: ty::Region<'tcx>,
            mutbl: hir::Mutability,
            s: &mut DiagStyledString,
        ) {
            s.push_highlighted(fmt_region(region));
            s.push_highlighted(mutbl.prefix_str());
        }

        fn maybe_highlight<T: Eq + ToString>(
            t1: T,
            t2: T,
            (buf1, buf2): &mut (DiagStyledString, DiagStyledString),
            tcx: TyCtxt<'_>,
        ) {
            let highlight = t1 != t2;
            let (t1, t2) = if highlight || tcx.sess.opts.verbose {
                (t1.to_string(), t2.to_string())
            } else {
                // The two types are the same, elide and don't highlight.
                ("_".into(), "_".into())
            };
            buf1.push(t1, highlight);
            buf2.push(t2, highlight);
        }

        fn cmp_ty_refs<'tcx>(
            r1: ty::Region<'tcx>,
            mut1: hir::Mutability,
            r2: ty::Region<'tcx>,
            mut2: hir::Mutability,
            ss: &mut (DiagStyledString, DiagStyledString),
        ) {
            let (r1, r2) = (fmt_region(r1), fmt_region(r2));
            if r1 != r2 {
                ss.0.push_highlighted(r1);
                ss.1.push_highlighted(r2);
            } else {
                ss.0.push_normal(r1);
                ss.1.push_normal(r2);
            }

            if mut1 != mut2 {
                ss.0.push_highlighted(mut1.prefix_str());
                ss.1.push_highlighted(mut2.prefix_str());
            } else {
                ss.0.push_normal(mut1.prefix_str());
                ss.1.push_normal(mut2.prefix_str());
            }
        }

        // process starts here
        match (t1.kind(), t2.kind()) {
            (&ty::Adt(def1, sub1), &ty::Adt(def2, sub2)) => {
                let did1 = def1.did();
                let did2 = def2.did();

                let generics1 = self.tcx.generics_of(did1);
                let generics2 = self.tcx.generics_of(did2);

                let non_default_after_default = generics1
                    .check_concrete_type_after_default(self.tcx, sub1)
                    || generics2.check_concrete_type_after_default(self.tcx, sub2);
                let sub_no_defaults_1 = if non_default_after_default {
                    generics1.own_args(sub1)
                } else {
                    generics1.own_args_no_defaults(self.tcx, sub1)
                };
                let sub_no_defaults_2 = if non_default_after_default {
                    generics2.own_args(sub2)
                } else {
                    generics2.own_args_no_defaults(self.tcx, sub2)
                };
                let mut values = (DiagStyledString::new(), DiagStyledString::new());
                let path1 = self.tcx.def_path_str(did1);
                let path2 = self.tcx.def_path_str(did2);
                if did1 == did2 {
                    // Easy case. Replace same types with `_` to shorten the output and highlight
                    // the differing ones.
                    //     let x: Foo<Bar, Qux> = y::<Foo<Quz, Qux>>();
                    //     Foo<Bar, _>
                    //     Foo<Quz, _>
                    //         ---  ^ type argument elided
                    //         |
                    //         highlighted in output
                    values.0.push_normal(path1);
                    values.1.push_normal(path2);

                    // Avoid printing out default generic parameters that are common to both
                    // types.
                    let len1 = sub_no_defaults_1.len();
                    let len2 = sub_no_defaults_2.len();
                    let common_len = cmp::min(len1, len2);
                    let remainder1 = &sub1[common_len..];
                    let remainder2 = &sub2[common_len..];
                    let common_default_params =
                        iter::zip(remainder1.iter().rev(), remainder2.iter().rev())
                            .filter(|(a, b)| a == b)
                            .count();
                    let len = sub1.len() - common_default_params;

                    // Only draw `<...>` if there are lifetime/type arguments.
                    if len > 0 {
                        values.0.push_normal("<");
                        values.1.push_normal("<");
                    }

                    fn lifetime_display(lifetime: Region<'_>) -> String {
                        let s = lifetime.to_string();
                        if s.is_empty() { "'_".to_string() } else { s }
                    }

                    for (i, (arg1, arg2)) in sub1.iter().zip(sub2).enumerate().take(len) {
                        self.push_comma(&mut values.0, &mut values.1, i);
                        match arg1.kind() {
                            // At one point we'd like to elide all lifetimes here, they are
                            // irrelevant for all diagnostics that use this output.
                            //
                            //     Foo<'x, '_, Bar>
                            //     Foo<'y, '_, Qux>
                            //         ^^  ^^  --- type arguments are not elided
                            //         |   |
                            //         |   elided as they were the same
                            //         not elided, they were different, but irrelevant
                            //
                            // For bound lifetimes, keep the names of the lifetimes,
                            // even if they are the same so that it's clear what's happening
                            // if we have something like
                            //
                            // for<'r, 's> fn(Inv<'r>, Inv<'s>)
                            // for<'r> fn(Inv<'r>, Inv<'r>)
                            ty::GenericArgKind::Lifetime(l1) => {
                                let l1_str = lifetime_display(l1);
                                let l2 = arg2.expect_region();
                                let l2_str = lifetime_display(l2);
                                if l1 != l2 {
                                    values.0.push_highlighted(l1_str);
                                    values.1.push_highlighted(l2_str);
                                } else if l1.is_bound() || self.tcx.sess.opts.verbose {
                                    values.0.push_normal(l1_str);
                                    values.1.push_normal(l2_str);
                                } else {
                                    values.0.push_normal("'_");
                                    values.1.push_normal("'_");
                                }
                            }
                            ty::GenericArgKind::Type(ta1) => {
                                let ta2 = arg2.expect_ty();
                                if ta1 == ta2 && !self.tcx.sess.opts.verbose {
                                    values.0.push_normal("_");
                                    values.1.push_normal("_");
                                } else {
                                    recurse(ta1, ta2, &mut values);
                                }
                            }
                            // We're comparing two types with the same path, so we compare the type
                            // arguments for both. If they are the same, do not highlight and elide
                            // from the output.
                            //     Foo<_, Bar>
                            //     Foo<_, Qux>
                            //         ^ elided type as this type argument was the same in both sides

                            // Do the same for const arguments, if they are equal, do not highlight and
                            // elide them from the output.
                            ty::GenericArgKind::Const(ca1) => {
                                let ca2 = arg2.expect_const();
                                maybe_highlight(ca1, ca2, &mut values, self.tcx);
                            }
                        }
                    }

                    // Close the type argument bracket.
                    // Only draw `<...>` if there are arguments.
                    if len > 0 {
                        values.0.push_normal(">");
                        values.1.push_normal(">");
                    }
                    values
                } else {
                    // Check for case:
                    //     let x: Foo<Bar<Qux> = foo::<Bar<Qux>>();
                    //     Foo<Bar<Qux>
                    //         ------- this type argument is exactly the same as the other type
                    //     Bar<Qux>
                    if self.cmp_type_arg(
                        &mut values.0,
                        &mut values.1,
                        path1.clone(),
                        sub_no_defaults_1,
                        path2.clone(),
                        t2,
                    ) {
                        return values;
                    }
                    // Check for case:
                    //     let x: Bar<Qux> = y:<Foo<Bar<Qux>>>();
                    //     Bar<Qux>
                    //     Foo<Bar<Qux>>
                    //         ------- this type argument is exactly the same as the other type
                    if self.cmp_type_arg(
                        &mut values.1,
                        &mut values.0,
                        path2,
                        sub_no_defaults_2,
                        path1,
                        t1,
                    ) {
                        return values;
                    }

                    // We can't find anything in common, highlight relevant part of type path.
                    //     let x: foo::bar::Baz<Qux> = y:<foo::bar::Bar<Zar>>();
                    //     foo::bar::Baz<Qux>
                    //     foo::bar::Bar<Zar>
                    //               -------- this part of the path is different

                    let t1_str = t1.to_string();
                    let t2_str = t2.to_string();
                    let min_len = t1_str.len().min(t2_str.len());

                    const SEPARATOR: &str = "::";
                    let separator_len = SEPARATOR.len();
                    let split_idx: usize =
                        iter::zip(t1_str.split(SEPARATOR), t2_str.split(SEPARATOR))
                            .take_while(|(mod1_str, mod2_str)| mod1_str == mod2_str)
                            .map(|(mod_str, _)| mod_str.len() + separator_len)
                            .sum();

                    debug!(?separator_len, ?split_idx, ?min_len, "cmp");

                    if split_idx >= min_len {
                        // paths are identical, highlight everything
                        (
                            DiagStyledString::highlighted(t1_str),
                            DiagStyledString::highlighted(t2_str),
                        )
                    } else {
                        let (common, uniq1) = t1_str.split_at(split_idx);
                        let (_, uniq2) = t2_str.split_at(split_idx);
                        debug!(?common, ?uniq1, ?uniq2, "cmp");

                        values.0.push_normal(common);
                        values.0.push_highlighted(uniq1);
                        values.1.push_normal(common);
                        values.1.push_highlighted(uniq2);

                        values
                    }
                }
            }

            // When finding `&T != &T`, compare the references, then recurse into pointee type
            (&ty::Ref(r1, ref_ty1, mutbl1), &ty::Ref(r2, ref_ty2, mutbl2)) => {
                let mut values = (DiagStyledString::new(), DiagStyledString::new());
                cmp_ty_refs(r1, mutbl1, r2, mutbl2, &mut values);
                recurse(ref_ty1, ref_ty2, &mut values);
                values
            }
            // When finding T != &T, highlight the borrow
            (&ty::Ref(r1, ref_ty1, mutbl1), _) => {
                let mut values = (DiagStyledString::new(), DiagStyledString::new());
                push_ref(r1, mutbl1, &mut values.0);
                recurse(ref_ty1, t2, &mut values);
                values
            }
            (_, &ty::Ref(r2, ref_ty2, mutbl2)) => {
                let mut values = (DiagStyledString::new(), DiagStyledString::new());
                push_ref(r2, mutbl2, &mut values.1);
                recurse(t1, ref_ty2, &mut values);
                values
            }

            // When encountering tuples of the same size, highlight only the differing types
            (&ty::Tuple(args1), &ty::Tuple(args2)) if args1.len() == args2.len() => {
                let mut values = (DiagStyledString::normal("("), DiagStyledString::normal("("));
                let len = args1.len();
                for (i, (left, right)) in args1.iter().zip(args2).enumerate() {
                    self.push_comma(&mut values.0, &mut values.1, i);
                    recurse(left, right, &mut values);
                }
                if len == 1 {
                    // Keep the output for single element tuples as `(ty,)`.
                    values.0.push_normal(",");
                    values.1.push_normal(",");
                }
                values.0.push_normal(")");
                values.1.push_normal(")");
                values
            }

            (ty::FnDef(did1, args1), ty::FnDef(did2, args2)) => {
                let sig1 = self.tcx.fn_sig(*did1).instantiate(self.tcx, args1);
                let sig2 = self.tcx.fn_sig(*did2).instantiate(self.tcx, args2);
                self.cmp_fn_sig(
                    &sig1,
                    Some((*did1, Some(args1))),
                    &sig2,
                    Some((*did2, Some(args2))),
                )
            }

            (ty::FnDef(did1, args1), ty::FnPtr(sig_tys2, hdr2)) => {
                let sig1 = self.tcx.fn_sig(*did1).instantiate(self.tcx, args1);
                self.cmp_fn_sig(&sig1, Some((*did1, Some(args1))), &sig_tys2.with(*hdr2), None)
            }

            (ty::FnPtr(sig_tys1, hdr1), ty::FnDef(did2, args2)) => {
                let sig2 = self.tcx.fn_sig(*did2).instantiate(self.tcx, args2);
                self.cmp_fn_sig(&sig_tys1.with(*hdr1), None, &sig2, Some((*did2, Some(args2))))
            }

            (ty::FnPtr(sig_tys1, hdr1), ty::FnPtr(sig_tys2, hdr2)) => {
                self.cmp_fn_sig(&sig_tys1.with(*hdr1), None, &sig_tys2.with(*hdr2), None)
            }

            _ => {
                let mut strs = (DiagStyledString::new(), DiagStyledString::new());
                maybe_highlight(t1, t2, &mut strs, self.tcx);
                strs
            }
        }
    }

    /// Extend a type error with extra labels pointing at "non-trivial" types, like closures and
    /// the return type of `async fn`s.
    ///
    /// `secondary_span` gives the caller the opportunity to expand `diag` with a `span_label`.
    ///
    /// `swap_secondary_and_primary` is used to make projection errors in particular nicer by using
    /// the message in `secondary_span` as the primary label, and apply the message that would
    /// otherwise be used for the primary label on the `secondary_span` `Span`. This applies on
    /// E0271, like `tests/ui/issues/issue-39970.stderr`.
    #[instrument(level = "debug", skip(self, diag, secondary_span, prefer_label))]
    pub fn note_type_err(
        &self,
        diag: &mut Diag<'_>,
        cause: &ObligationCause<'tcx>,
        secondary_span: Option<(Span, Cow<'static, str>, bool)>,
        mut values: Option<ty::ParamEnvAnd<'tcx, ValuePairs<'tcx>>>,
        terr: TypeError<'tcx>,
        prefer_label: bool,
        override_span: Option<Span>,
    ) {
        // We use `override_span` when we want the error to point at a `Span` other than
        // `cause.span`. This is used in E0271, when a closure is passed in where the return type
        // isn't what was expected. We want to point at the closure's return type (or expression),
        // instead of the expression where the closure is passed as call argument.
        let span = override_span.unwrap_or(cause.span);
        // For some types of errors, expected-found does not make
        // sense, so just ignore the values we were given.
        if let TypeError::CyclicTy(_) = terr {
            values = None;
        }
        struct OpaqueTypesVisitor<'tcx> {
            types: FxIndexMap<TyCategory, FxIndexSet<Span>>,
            expected: FxIndexMap<TyCategory, FxIndexSet<Span>>,
            found: FxIndexMap<TyCategory, FxIndexSet<Span>>,
            ignore_span: Span,
            tcx: TyCtxt<'tcx>,
        }

        impl<'tcx> OpaqueTypesVisitor<'tcx> {
            fn visit_expected_found(
                tcx: TyCtxt<'tcx>,
                expected: impl TypeVisitable<TyCtxt<'tcx>>,
                found: impl TypeVisitable<TyCtxt<'tcx>>,
                ignore_span: Span,
            ) -> Self {
                let mut types_visitor = OpaqueTypesVisitor {
                    types: Default::default(),
                    expected: Default::default(),
                    found: Default::default(),
                    ignore_span,
                    tcx,
                };
                // The visitor puts all the relevant encountered types in `self.types`, but in
                // here we want to visit two separate types with no relation to each other, so we
                // move the results from `types` to `expected` or `found` as appropriate.
                expected.visit_with(&mut types_visitor);
                std::mem::swap(&mut types_visitor.expected, &mut types_visitor.types);
                found.visit_with(&mut types_visitor);
                std::mem::swap(&mut types_visitor.found, &mut types_visitor.types);
                types_visitor
            }

            fn report(&self, err: &mut Diag<'_>) {
                self.add_labels_for_types(err, "expected", &self.expected);
                self.add_labels_for_types(err, "found", &self.found);
            }

            fn add_labels_for_types(
                &self,
                err: &mut Diag<'_>,
                target: &str,
                types: &FxIndexMap<TyCategory, FxIndexSet<Span>>,
            ) {
                for (kind, values) in types.iter() {
                    let count = values.len();
                    for &sp in values {
                        err.span_label(
                            sp,
                            format!(
                                "{}{} {:#}{}",
                                if count == 1 { "the " } else { "one of the " },
                                target,
                                kind,
                                pluralize!(count),
                            ),
                        );
                    }
                }
            }
        }

        impl<'tcx> ty::TypeVisitor<TyCtxt<'tcx>> for OpaqueTypesVisitor<'tcx> {
            fn visit_ty(&mut self, t: Ty<'tcx>) {
                if let Some((kind, def_id)) = TyCategory::from_ty(self.tcx, t) {
                    let span = self.tcx.def_span(def_id);
                    // Avoid cluttering the output when the "found" and error span overlap:
                    //
                    // error[E0308]: mismatched types
                    //   --> $DIR/issue-20862.rs:2:5
                    //    |
                    // LL |     |y| x + y
                    //    |     ^^^^^^^^^
                    //    |     |
                    //    |     the found closure
                    //    |     expected `()`, found closure
                    //    |
                    //    = note: expected unit type `()`
                    //                 found closure `{closure@$DIR/issue-20862.rs:2:5: 2:14 x:_}`
                    //
                    // Also ignore opaque `Future`s that come from async fns.
                    if !self.ignore_span.overlaps(span)
                        && !span.is_desugaring(DesugaringKind::Async)
                    {
                        self.types.entry(kind).or_default().insert(span);
                    }
                }
                t.super_visit_with(self)
            }
        }

        debug!("note_type_err(diag={:?})", diag);
        enum Mismatch<'a> {
            Variable(ty::error::ExpectedFound<Ty<'a>>),
            Fixed(&'static str),
        }
        let (expected_found, exp_found, is_simple_error, values, param_env) = match values {
            None => (None, Mismatch::Fixed("type"), false, None, None),
            Some(ty::ParamEnvAnd { param_env, value: values }) => {
                let mut values = self.resolve_vars_if_possible(values);
                if self.next_trait_solver() {
                    values = deeply_normalize_for_diagnostics(self, param_env, values);
                }
                let (is_simple_error, exp_found) = match values {
                    ValuePairs::Terms(ExpectedFound { expected, found }) => {
                        match (expected.kind(), found.kind()) {
                            (ty::TermKind::Ty(expected), ty::TermKind::Ty(found)) => {
                                let is_simple_err =
                                    expected.is_simple_text() && found.is_simple_text();
                                OpaqueTypesVisitor::visit_expected_found(
                                    self.tcx, expected, found, span,
                                )
                                .report(diag);

                                (
                                    is_simple_err,
                                    Mismatch::Variable(ExpectedFound { expected, found }),
                                )
                            }
                            (ty::TermKind::Const(_), ty::TermKind::Const(_)) => {
                                (false, Mismatch::Fixed("constant"))
                            }
                            _ => (false, Mismatch::Fixed("type")),
                        }
                    }
                    ValuePairs::PolySigs(ExpectedFound { expected, found }) => {
                        OpaqueTypesVisitor::visit_expected_found(self.tcx, expected, found, span)
                            .report(diag);
                        (false, Mismatch::Fixed("signature"))
                    }
                    ValuePairs::TraitRefs(_) => (false, Mismatch::Fixed("trait")),
                    ValuePairs::Aliases(ExpectedFound { expected, .. }) => {
                        (false, Mismatch::Fixed(self.tcx.def_descr(expected.def_id)))
                    }
                    ValuePairs::Regions(_) => (false, Mismatch::Fixed("lifetime")),
                    ValuePairs::ExistentialTraitRef(_) => {
                        (false, Mismatch::Fixed("existential trait ref"))
                    }
                    ValuePairs::ExistentialProjection(_) => {
                        (false, Mismatch::Fixed("existential projection"))
                    }
                };
                let Some(vals) = self.values_str(values, cause, diag.long_ty_path()) else {
                    // Derived error. Cancel the emitter.
                    // NOTE(eddyb) this was `.cancel()`, but `diag`
                    // is borrowed, so we can't fully defuse it.
                    diag.downgrade_to_delayed_bug();
                    return;
                };
                (Some(vals), exp_found, is_simple_error, Some(values), Some(param_env))
            }
        };

        let mut label_or_note = |span: Span, msg: Cow<'static, str>| {
            if (prefer_label && is_simple_error) || &[span] == diag.span.primary_spans() {
                diag.span_label(span, msg);
            } else {
                diag.span_note(span, msg);
            }
        };
        if let Some((secondary_span, secondary_msg, swap_secondary_and_primary)) = secondary_span {
            if swap_secondary_and_primary {
                let terr = if let Some(infer::ValuePairs::Terms(ExpectedFound {
                    expected, ..
                })) = values
                {
                    Cow::from(format!("expected this to be `{expected}`"))
                } else {
                    terr.to_string(self.tcx)
                };
                label_or_note(secondary_span, terr);
                label_or_note(span, secondary_msg);
            } else {
                label_or_note(span, terr.to_string(self.tcx));
                label_or_note(secondary_span, secondary_msg);
            }
        } else if let Some(values) = values
            && let Some((e, f)) = values.ty()
            && let TypeError::ArgumentSorts(..) | TypeError::Sorts(_) = terr
        {
            let e = self.tcx.erase_regions(e);
            let f = self.tcx.erase_regions(f);
            let expected = with_forced_trimmed_paths!(e.sort_string(self.tcx));
            let found = with_forced_trimmed_paths!(f.sort_string(self.tcx));
            if expected == found {
                label_or_note(span, terr.to_string(self.tcx));
            } else {
                label_or_note(span, Cow::from(format!("expected {expected}, found {found}")));
            }
        } else {
            label_or_note(span, terr.to_string(self.tcx));
        }

        if self.check_and_note_conflicting_crates(diag, terr) {
            return;
        }

        if let Some((expected, found)) = expected_found {
            let (expected_label, found_label, exp_found) = match exp_found {
                Mismatch::Variable(ef) => (
                    ef.expected.prefix_string(self.tcx),
                    ef.found.prefix_string(self.tcx),
                    Some(ef),
                ),
                Mismatch::Fixed(s) => (s.into(), s.into(), None),
            };

            enum Similar<'tcx> {
                Adts { expected: ty::AdtDef<'tcx>, found: ty::AdtDef<'tcx> },
                PrimitiveFound { expected: ty::AdtDef<'tcx>, found: Ty<'tcx> },
                PrimitiveExpected { expected: Ty<'tcx>, found: ty::AdtDef<'tcx> },
            }

            let similarity = |ExpectedFound { expected, found }: ExpectedFound<Ty<'tcx>>| {
                if let ty::Adt(expected, _) = expected.kind()
                    && let Some(primitive) = found.primitive_symbol()
                {
                    let path = self.tcx.def_path(expected.did()).data;
                    let name = path.last().unwrap().data.get_opt_name();
                    if name == Some(primitive) {
                        return Some(Similar::PrimitiveFound { expected: *expected, found });
                    }
                } else if let Some(primitive) = expected.primitive_symbol()
                    && let ty::Adt(found, _) = found.kind()
                {
                    let path = self.tcx.def_path(found.did()).data;
                    let name = path.last().unwrap().data.get_opt_name();
                    if name == Some(primitive) {
                        return Some(Similar::PrimitiveExpected { expected, found: *found });
                    }
                } else if let ty::Adt(expected, _) = expected.kind()
                    && let ty::Adt(found, _) = found.kind()
                {
                    if !expected.did().is_local() && expected.did().krate == found.did().krate {
                        // Most likely types from different versions of the same crate
                        // are in play, in which case this message isn't so helpful.
                        // A "perhaps two different versions..." error is already emitted for that.
                        return None;
                    }
                    let f_path = self.tcx.def_path(found.did()).data;
                    let e_path = self.tcx.def_path(expected.did()).data;

                    if let (Some(e_last), Some(f_last)) = (e_path.last(), f_path.last())
                        && e_last == f_last
                    {
                        return Some(Similar::Adts { expected: *expected, found: *found });
                    }
                }
                None
            };

            match terr {
                // If two types mismatch but have similar names, mention that specifically.
                TypeError::Sorts(values) if let Some(s) = similarity(values) => {
                    let diagnose_primitive =
                        |prim: Ty<'tcx>, shadow: Ty<'tcx>, defid: DefId, diag: &mut Diag<'_>| {
                            let name = shadow.sort_string(self.tcx);
                            diag.note(format!(
                                "`{prim}` and {name} have similar names, but are actually distinct types"
                            ));
                            diag.note(format!(
                                "one `{prim}` is a primitive defined by the language",
                            ));
                            let def_span = self.tcx.def_span(defid);
                            let msg = if defid.is_local() {
                                format!("the other {name} is defined in the current crate")
                            } else {
                                let crate_name = self.tcx.crate_name(defid.krate);
                                format!("the other {name} is defined in crate `{crate_name}`")
                            };
                            diag.span_note(def_span, msg);
                        };

                    let diagnose_adts =
                        |expected_adt: ty::AdtDef<'tcx>,
                         found_adt: ty::AdtDef<'tcx>,
                         diag: &mut Diag<'_>| {
                            let found_name = values.found.sort_string(self.tcx);
                            let expected_name = values.expected.sort_string(self.tcx);

                            let found_defid = found_adt.did();
                            let expected_defid = expected_adt.did();

                            diag.note(format!("{found_name} and {expected_name} have similar names, but are actually distinct types"));
                            for (defid, name) in
                                [(found_defid, found_name), (expected_defid, expected_name)]
                            {
                                let def_span = self.tcx.def_span(defid);

                                let msg = if found_defid.is_local() && expected_defid.is_local() {
                                    let module = self
                                        .tcx
                                        .parent_module_from_def_id(defid.expect_local())
                                        .to_def_id();
                                    let module_name =
                                        self.tcx.def_path(module).to_string_no_crate_verbose();
                                    format!(
                                        "{name} is defined in module `crate{module_name}` of the current crate"
                                    )
                                } else if defid.is_local() {
                                    format!("{name} is defined in the current crate")
                                } else {
                                    let crate_name = self.tcx.crate_name(defid.krate);
                                    format!("{name} is defined in crate `{crate_name}`")
                                };
                                diag.span_note(def_span, msg);
                            }
                        };

                    match s {
                        Similar::Adts { expected, found } => diagnose_adts(expected, found, diag),
                        Similar::PrimitiveFound { expected, found: prim } => {
                            diagnose_primitive(prim, values.expected, expected.did(), diag)
                        }
                        Similar::PrimitiveExpected { expected: prim, found } => {
                            diagnose_primitive(prim, values.found, found.did(), diag)
                        }
                    }
                }
                TypeError::Sorts(values) => {
                    let extra = expected == found
                        // Ensure that we don't ever say something like
                        // expected `impl Trait` (opaque type `impl Trait`)
                        //    found `impl Trait` (opaque type `impl Trait`)
                        && values.expected.sort_string(self.tcx)
                            != values.found.sort_string(self.tcx);
                    let sort_string = |ty: Ty<'tcx>| match (extra, ty.kind()) {
                        (true, ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. })) => {
                            let sm = self.tcx.sess.source_map();
                            let pos = sm.lookup_char_pos(self.tcx.def_span(*def_id).lo());
                            DiagStyledString::normal(format!(
                                " (opaque type at <{}:{}:{}>)",
                                sm.filename_for_diagnostics(&pos.file.name),
                                pos.line,
                                pos.col.to_usize() + 1,
                            ))
                        }
                        (true, ty::Alias(ty::Projection, proj))
                            if self.tcx.is_impl_trait_in_trait(proj.def_id) =>
                        {
                            let sm = self.tcx.sess.source_map();
                            let pos = sm.lookup_char_pos(self.tcx.def_span(proj.def_id).lo());
                            DiagStyledString::normal(format!(
                                " (trait associated opaque type at <{}:{}:{}>)",
                                sm.filename_for_diagnostics(&pos.file.name),
                                pos.line,
                                pos.col.to_usize() + 1,
                            ))
                        }
                        (true, _) => {
                            let mut s = DiagStyledString::normal(" (");
                            s.push_highlighted(ty.sort_string(self.tcx));
                            s.push_normal(")");
                            s
                        }
                        (false, _) => DiagStyledString::normal(""),
                    };
                    if !(values.expected.is_simple_text() && values.found.is_simple_text())
                        || (exp_found.is_some_and(|ef| {
                            // This happens when the type error is a subset of the expectation,
                            // like when you have two references but one is `usize` and the other
                            // is `f32`. In those cases we still want to show the `note`. If the
                            // value from `ef` is `Infer(_)`, then we ignore it.
                            if !ef.expected.is_ty_or_numeric_infer() {
                                ef.expected != values.expected
                            } else if !ef.found.is_ty_or_numeric_infer() {
                                ef.found != values.found
                            } else {
                                false
                            }
                        }))
                    {
                        if let Some(ExpectedFound { found: found_ty, .. }) = exp_found
                            && !self.tcx.ty_is_opaque_future(found_ty)
                        {
                            // `Future` is a special opaque type that the compiler
                            // will try to hide in some case such as `async fn`, so
                            // to make an error more use friendly we will
                            // avoid to suggest a mismatch type with a
                            // type that the user usually are not using
                            // directly such as `impl Future<Output = u8>`.
                            diag.note_expected_found_extra(
                                &expected_label,
                                expected,
                                &found_label,
                                found,
                                sort_string(values.expected),
                                sort_string(values.found),
                            );
                        }
                    }
                }
                _ => {
                    debug!(
                        "note_type_err: exp_found={:?}, expected={:?} found={:?}",
                        exp_found, expected, found
                    );
                    if !is_simple_error || terr.must_include_note() {
                        diag.note_expected_found(&expected_label, expected, &found_label, found);

                        if let Some(ty::Closure(_, args)) =
                            exp_found.map(|expected_type_found| expected_type_found.found.kind())
                        {
                            diag.highlighted_note(vec![
                                StringPart::normal("closure has signature: `"),
                                StringPart::highlighted(
                                    self.tcx
                                        .signature_unclosure(
                                            args.as_closure().sig(),
                                            rustc_hir::Safety::Safe,
                                        )
                                        .to_string(),
                                ),
                                StringPart::normal("`"),
                            ]);
                        }
                    }
                }
            }
        }
        let exp_found = match exp_found {
            Mismatch::Variable(exp_found) => Some(exp_found),
            Mismatch::Fixed(_) => None,
        };
        let exp_found = match terr {
            // `terr` has more accurate type information than `exp_found` in match expressions.
            ty::error::TypeError::Sorts(terr)
                if exp_found.is_some_and(|ef| terr.found == ef.found) =>
            {
                Some(terr)
            }
            _ => exp_found,
        };
        debug!("exp_found {:?} terr {:?} cause.code {:?}", exp_found, terr, cause.code());
        if let Some(exp_found) = exp_found {
            let should_suggest_fixes =
                if let ObligationCauseCode::Pattern { root_ty, .. } = cause.code() {
                    // Skip if the root_ty of the pattern is not the same as the expected_ty.
                    // If these types aren't equal then we've probably peeled off a layer of arrays.
                    self.same_type_modulo_infer(*root_ty, exp_found.expected)
                } else {
                    true
                };

            // FIXME(#73154): For now, we do leak check when coercing function
            // pointers in typeck, instead of only during borrowck. This can lead
            // to these `RegionsInsufficientlyPolymorphic` errors that aren't helpful.
            if should_suggest_fixes
                && !matches!(terr, TypeError::RegionsInsufficientlyPolymorphic(..))
            {
                self.suggest_tuple_pattern(cause, &exp_found, diag);
                self.suggest_accessing_field_where_appropriate(cause, &exp_found, diag);
                self.suggest_await_on_expect_found(cause, span, &exp_found, diag);
                self.suggest_function_pointers(cause, span, &exp_found, terr, diag);
                self.suggest_turning_stmt_into_expr(cause, &exp_found, diag);
            }
        }

        self.note_and_explain_type_err(diag, terr, cause, span, cause.body_id.to_def_id());
        if let Some(exp_found) = exp_found
            && let exp_found = TypeError::Sorts(exp_found)
            && exp_found != terr
        {
            self.note_and_explain_type_err(diag, exp_found, cause, span, cause.body_id.to_def_id());
        }

        if let Some(ValuePairs::TraitRefs(exp_found)) = values
            && let ty::Closure(def_id, _) = exp_found.expected.self_ty().kind()
            && let Some(def_id) = def_id.as_local()
            && terr.involves_regions()
        {
            let span = self.tcx.def_span(def_id);
            diag.span_note(span, "this closure does not fulfill the lifetime requirements");
            self.suggest_for_all_lifetime_closure(
                span,
                self.tcx.hir_node_by_def_id(def_id),
                &exp_found,
                diag,
            );
        }

        // It reads better to have the error origin as the final
        // thing.
        self.note_error_origin(diag, cause, exp_found, terr, param_env);

        debug!(?diag);
    }

    pub fn type_error_additional_suggestions(
        &self,
        trace: &TypeTrace<'tcx>,
        terr: TypeError<'tcx>,
        path: &mut Option<PathBuf>,
    ) -> Vec<TypeErrorAdditionalDiags> {
        let mut suggestions = Vec::new();
        let span = trace.cause.span;
        let values = self.resolve_vars_if_possible(trace.values);
        if let Some((expected, found)) = values.ty() {
            match (expected.kind(), found.kind()) {
                (ty::Tuple(_), ty::Tuple(_)) => {}
                // If a tuple of length one was expected and the found expression has
                // parentheses around it, perhaps the user meant to write `(expr,)` to
                // build a tuple (issue #86100)
                (ty::Tuple(fields), _) => {
                    suggestions.extend(self.suggest_wrap_to_build_a_tuple(span, found, fields))
                }
                // If a byte was expected and the found expression is a char literal
                // containing a single ASCII character, perhaps the user meant to write `b'c'` to
                // specify a byte literal
                (ty::Uint(ty::UintTy::U8), ty::Char) => {
                    if let Ok(code) = self.tcx.sess().source_map().span_to_snippet(span)
                        && let Some(code) =
                            code.strip_prefix('\'').and_then(|s| s.strip_suffix('\''))
                        // forbid all Unicode escapes
                        && !code.starts_with("\\u")
                        // forbids literal Unicode characters beyond ASCII
                        && code.chars().next().is_some_and(|c| c.is_ascii())
                    {
                        suggestions.push(TypeErrorAdditionalDiags::MeantByteLiteral {
                            span,
                            code: escape_literal(code),
                        })
                    }
                }
                // If a character was expected and the found expression is a string literal
                // containing a single character, perhaps the user meant to write `'c'` to
                // specify a character literal (issue #92479)
                (ty::Char, ty::Ref(_, r, _)) if r.is_str() => {
                    if let Ok(code) = self.tcx.sess().source_map().span_to_snippet(span)
                        && let Some(code) = code.strip_prefix('"').and_then(|s| s.strip_suffix('"'))
                        && code.chars().count() == 1
                    {
                        suggestions.push(TypeErrorAdditionalDiags::MeantCharLiteral {
                            span,
                            code: escape_literal(code),
                        })
                    }
                }
                // If a string was expected and the found expression is a character literal,
                // perhaps the user meant to write `"s"` to specify a string literal.
                (ty::Ref(_, r, _), ty::Char) if r.is_str() => {
                    if let Ok(code) = self.tcx.sess().source_map().span_to_snippet(span)
                        && code.starts_with("'")
                        && code.ends_with("'")
                    {
                        suggestions.push(TypeErrorAdditionalDiags::MeantStrLiteral {
                            start: span.with_hi(span.lo() + BytePos(1)),
                            end: span.with_lo(span.hi() - BytePos(1)),
                        });
                    }
                }
                // For code `if Some(..) = expr `, the type mismatch may be expected `bool` but found `()`,
                // we try to suggest to add the missing `let` for `if let Some(..) = expr`
                (ty::Bool, ty::Tuple(list)) => {
                    if list.len() == 0 {
                        suggestions.extend(self.suggest_let_for_letchains(&trace.cause, span));
                    }
                }
                (ty::Array(_, _), ty::Array(_, _)) => {
                    suggestions.extend(self.suggest_specify_actual_length(terr, trace, span))
                }
                _ => {}
            }
        }
        let code = trace.cause.code();
        if let &(ObligationCauseCode::MatchExpressionArm(box MatchExpressionArmCause {
            source,
            ..
        })
        | ObligationCauseCode::BlockTailExpression(.., source)) = code
            && let hir::MatchSource::TryDesugar(_) = source
            && let Some((expected_ty, found_ty)) = self.values_str(trace.values, &trace.cause, path)
        {
            suggestions.push(TypeErrorAdditionalDiags::TryCannotConvert {
                found: found_ty.content(),
                expected: expected_ty.content(),
            });
        }
        suggestions
    }

    fn suggest_specify_actual_length(
        &self,
        terr: TypeError<'tcx>,
        trace: &TypeTrace<'tcx>,
        span: Span,
    ) -> Option<TypeErrorAdditionalDiags> {
        let TypeError::ArraySize(sz) = terr else {
            return None;
        };
        let tykind = match self.tcx.hir_node_by_def_id(trace.cause.body_id) {
            hir::Node::Item(hir::Item {
                kind: hir::ItemKind::Fn { body: body_id, .. }, ..
            }) => {
                let body = self.tcx.hir_body(*body_id);
                struct LetVisitor {
                    span: Span,
                }
                impl<'v> Visitor<'v> for LetVisitor {
                    type Result = ControlFlow<&'v hir::TyKind<'v>>;
                    fn visit_stmt(&mut self, s: &'v hir::Stmt<'v>) -> Self::Result {
                        // Find a local statement where the initializer has
                        // the same span as the error and the type is specified.
                        if let hir::Stmt {
                            kind:
                                hir::StmtKind::Let(hir::LetStmt {
                                    init: Some(hir::Expr { span: init_span, .. }),
                                    ty: Some(array_ty),
                                    ..
                                }),
                            ..
                        } = s
                            && init_span == &self.span
                        {
                            ControlFlow::Break(&array_ty.peel_refs().kind)
                        } else {
                            ControlFlow::Continue(())
                        }
                    }
                }
                LetVisitor { span }.visit_body(body).break_value()
            }
            hir::Node::Item(hir::Item { kind: hir::ItemKind::Const(_, _, ty, _), .. }) => {
                Some(&ty.peel_refs().kind)
            }
            _ => None,
        };
        if let Some(tykind) = tykind
            && let hir::TyKind::Array(_, length_arg) = tykind
            && let Some(length_val) = sz.found.try_to_target_usize(self.tcx)
        {
            Some(TypeErrorAdditionalDiags::ConsiderSpecifyingLength {
                span: length_arg.span(),
                length: length_val,
            })
        } else {
            None
        }
    }

    pub fn report_and_explain_type_error(
        &self,
        trace: TypeTrace<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        terr: TypeError<'tcx>,
    ) -> Diag<'a> {
        debug!("report_and_explain_type_error(trace={:?}, terr={:?})", trace, terr);

        let span = trace.cause.span;
        let mut path = None;
        let failure_code = trace.cause.as_failure_code_diag(
            terr,
            span,
            self.type_error_additional_suggestions(&trace, terr, &mut path),
        );
        let mut diag = self.dcx().create_err(failure_code);
        *diag.long_ty_path() = path;
        self.note_type_err(
            &mut diag,
            &trace.cause,
            None,
            Some(param_env.and(trace.values)),
            terr,
            false,
            None,
        );
        diag
    }

    fn suggest_wrap_to_build_a_tuple(
        &self,
        span: Span,
        found: Ty<'tcx>,
        expected_fields: &List<Ty<'tcx>>,
    ) -> Option<TypeErrorAdditionalDiags> {
        let [expected_tup_elem] = expected_fields[..] else { return None };

        if !self.same_type_modulo_infer(expected_tup_elem, found) {
            return None;
        }

        let Ok(code) = self.tcx.sess().source_map().span_to_snippet(span) else { return None };

        let sugg = if code.starts_with('(') && code.ends_with(')') {
            let before_close = span.hi() - BytePos::from_u32(1);
            TypeErrorAdditionalDiags::TupleOnlyComma {
                span: span.with_hi(before_close).shrink_to_hi(),
            }
        } else {
            TypeErrorAdditionalDiags::TupleAlsoParentheses {
                span_low: span.shrink_to_lo(),
                span_high: span.shrink_to_hi(),
            }
        };
        Some(sugg)
    }

    fn values_str(
        &self,
        values: ValuePairs<'tcx>,
        cause: &ObligationCause<'tcx>,
        file: &mut Option<PathBuf>,
    ) -> Option<(DiagStyledString, DiagStyledString)> {
        match values {
            ValuePairs::Regions(exp_found) => self.expected_found_str(exp_found),
            ValuePairs::Terms(exp_found) => self.expected_found_str_term(exp_found, file),
            ValuePairs::Aliases(exp_found) => self.expected_found_str(exp_found),
            ValuePairs::ExistentialTraitRef(exp_found) => self.expected_found_str(exp_found),
            ValuePairs::ExistentialProjection(exp_found) => self.expected_found_str(exp_found),
            ValuePairs::TraitRefs(exp_found) => {
                let pretty_exp_found = ty::error::ExpectedFound {
                    expected: exp_found.expected.print_trait_sugared(),
                    found: exp_found.found.print_trait_sugared(),
                };
                match self.expected_found_str(pretty_exp_found) {
                    Some((expected, found)) if expected == found => {
                        self.expected_found_str(exp_found)
                    }
                    ret => ret,
                }
            }
            ValuePairs::PolySigs(exp_found) => {
                let exp_found = self.resolve_vars_if_possible(exp_found);
                if exp_found.references_error() {
                    return None;
                }
                let (fn_def1, fn_def2) = if let ObligationCauseCode::CompareImplItem {
                    impl_item_def_id,
                    trait_item_def_id,
                    ..
                } = *cause.code()
                {
                    (Some((trait_item_def_id, None)), Some((impl_item_def_id.to_def_id(), None)))
                } else {
                    (None, None)
                };

                Some(self.cmp_fn_sig(&exp_found.expected, fn_def1, &exp_found.found, fn_def2))
            }
        }
    }

    fn expected_found_str_term(
        &self,
        exp_found: ty::error::ExpectedFound<ty::Term<'tcx>>,
        path: &mut Option<PathBuf>,
    ) -> Option<(DiagStyledString, DiagStyledString)> {
        let exp_found = self.resolve_vars_if_possible(exp_found);
        if exp_found.references_error() {
            return None;
        }

        Some(match (exp_found.expected.kind(), exp_found.found.kind()) {
            (ty::TermKind::Ty(expected), ty::TermKind::Ty(found)) => {
                let (mut exp, mut fnd) = self.cmp(expected, found);
                // Use the terminal width as the basis to determine when to compress the printed
                // out type, but give ourselves some leeway to avoid ending up creating a file for
                // a type that is somewhat shorter than the path we'd write to.
                let len = self.tcx.sess().diagnostic_width() + 40;
                let exp_s = exp.content();
                let fnd_s = fnd.content();
                if exp_s.len() > len {
                    let exp_s = self.tcx.short_string(expected, path);
                    exp = DiagStyledString::highlighted(exp_s);
                }
                if fnd_s.len() > len {
                    let fnd_s = self.tcx.short_string(found, path);
                    fnd = DiagStyledString::highlighted(fnd_s);
                }
                (exp, fnd)
            }
            _ => (
                DiagStyledString::highlighted(exp_found.expected.to_string()),
                DiagStyledString::highlighted(exp_found.found.to_string()),
            ),
        })
    }

    /// Returns a string of the form "expected `{}`, found `{}`".
    fn expected_found_str<T: fmt::Display + TypeFoldable<TyCtxt<'tcx>>>(
        &self,
        exp_found: ty::error::ExpectedFound<T>,
    ) -> Option<(DiagStyledString, DiagStyledString)> {
        let exp_found = self.resolve_vars_if_possible(exp_found);
        if exp_found.references_error() {
            return None;
        }

        Some((
            DiagStyledString::highlighted(exp_found.expected.to_string()),
            DiagStyledString::highlighted(exp_found.found.to_string()),
        ))
    }

    /// Determine whether an error associated with the given span and definition
    /// should be treated as being caused by the implicit `From` conversion
    /// within `?` desugaring.
    pub fn is_try_conversion(&self, span: Span, trait_def_id: DefId) -> bool {
        span.is_desugaring(DesugaringKind::QuestionMark)
            && self.tcx.is_diagnostic_item(sym::From, trait_def_id)
    }

    /// Structurally compares two types, modulo any inference variables.
    ///
    /// Returns `true` if two types are equal, or if one type is an inference variable compatible
    /// with the other type. A TyVar inference type is compatible with any type, and an IntVar or
    /// FloatVar inference type are compatible with themselves or their concrete types (Int and
    /// Float types, respectively). When comparing two ADTs, these rules apply recursively.
    pub fn same_type_modulo_infer<T: relate::Relate<TyCtxt<'tcx>>>(&self, a: T, b: T) -> bool {
        let (a, b) = self.resolve_vars_if_possible((a, b));
        SameTypeModuloInfer(self).relate(a, b).is_ok()
    }
}

struct SameTypeModuloInfer<'a, 'tcx>(&'a InferCtxt<'tcx>);

impl<'tcx> TypeRelation<TyCtxt<'tcx>> for SameTypeModuloInfer<'_, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.0.tcx
    }

    fn relate_with_variance<T: relate::Relate<TyCtxt<'tcx>>>(
        &mut self,
        _variance: ty::Variance,
        _info: ty::VarianceDiagInfo<TyCtxt<'tcx>>,
        a: T,
        b: T,
    ) -> relate::RelateResult<'tcx, T> {
        self.relate(a, b)
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        match (a.kind(), b.kind()) {
            (ty::Int(_) | ty::Uint(_), ty::Infer(ty::InferTy::IntVar(_)))
            | (
                ty::Infer(ty::InferTy::IntVar(_)),
                ty::Int(_) | ty::Uint(_) | ty::Infer(ty::InferTy::IntVar(_)),
            )
            | (ty::Float(_), ty::Infer(ty::InferTy::FloatVar(_)))
            | (
                ty::Infer(ty::InferTy::FloatVar(_)),
                ty::Float(_) | ty::Infer(ty::InferTy::FloatVar(_)),
            )
            | (ty::Infer(ty::InferTy::TyVar(_)), _)
            | (_, ty::Infer(ty::InferTy::TyVar(_))) => Ok(a),
            (ty::Infer(_), _) | (_, ty::Infer(_)) => Err(TypeError::Mismatch),
            _ => relate::structurally_relate_tys(self, a, b),
        }
    }

    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        if (a.is_var() && b.is_free())
            || (b.is_var() && a.is_free())
            || (a.is_var() && b.is_var())
            || a == b
        {
            Ok(a)
        } else {
            Err(TypeError::Mismatch)
        }
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        b: ty::Binder<'tcx, T>,
    ) -> relate::RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: relate::Relate<TyCtxt<'tcx>>,
    {
        Ok(a.rebind(self.relate(a.skip_binder(), b.skip_binder())?))
    }

    fn consts(
        &mut self,
        a: ty::Const<'tcx>,
        _b: ty::Const<'tcx>,
    ) -> relate::RelateResult<'tcx, ty::Const<'tcx>> {
        // FIXME(compiler-errors): This could at least do some first-order
        // relation
        Ok(a)
    }
}

pub enum FailureCode {
    Error0317,
    Error0580,
    Error0308,
    Error0644,
}

#[extension(pub trait ObligationCauseExt<'tcx>)]
impl<'tcx> ObligationCause<'tcx> {
    fn as_failure_code(&self, terr: TypeError<'tcx>) -> FailureCode {
        match self.code() {
            ObligationCauseCode::IfExpressionWithNoElse => FailureCode::Error0317,
            ObligationCauseCode::MainFunctionType => FailureCode::Error0580,
            ObligationCauseCode::CompareImplItem { .. }
            | ObligationCauseCode::MatchExpressionArm(_)
            | ObligationCauseCode::IfExpression { .. }
            | ObligationCauseCode::LetElse
            | ObligationCauseCode::LangFunctionType(_)
            | ObligationCauseCode::IntrinsicType
            | ObligationCauseCode::MethodReceiver => FailureCode::Error0308,

            // In the case where we have no more specific thing to
            // say, also take a look at the error code, maybe we can
            // tailor to that.
            _ => match terr {
                TypeError::CyclicTy(ty)
                    if ty.is_closure() || ty.is_coroutine() || ty.is_coroutine_closure() =>
                {
                    FailureCode::Error0644
                }
                TypeError::IntrinsicCast | TypeError::ForceInlineCast => FailureCode::Error0308,
                _ => FailureCode::Error0308,
            },
        }
    }
    fn as_failure_code_diag(
        &self,
        terr: TypeError<'tcx>,
        span: Span,
        subdiags: Vec<TypeErrorAdditionalDiags>,
    ) -> ObligationCauseFailureCode {
        match self.code() {
            ObligationCauseCode::CompareImplItem { kind: ty::AssocKind::Fn { .. }, .. } => {
                ObligationCauseFailureCode::MethodCompat { span, subdiags }
            }
            ObligationCauseCode::CompareImplItem { kind: ty::AssocKind::Type { .. }, .. } => {
                ObligationCauseFailureCode::TypeCompat { span, subdiags }
            }
            ObligationCauseCode::CompareImplItem { kind: ty::AssocKind::Const { .. }, .. } => {
                ObligationCauseFailureCode::ConstCompat { span, subdiags }
            }
            ObligationCauseCode::BlockTailExpression(.., hir::MatchSource::TryDesugar(_)) => {
                ObligationCauseFailureCode::TryCompat { span, subdiags }
            }
            ObligationCauseCode::MatchExpressionArm(box MatchExpressionArmCause {
                source, ..
            }) => match source {
                hir::MatchSource::TryDesugar(_) => {
                    ObligationCauseFailureCode::TryCompat { span, subdiags }
                }
                _ => ObligationCauseFailureCode::MatchCompat { span, subdiags },
            },
            ObligationCauseCode::IfExpression { .. } => {
                ObligationCauseFailureCode::IfElseDifferent { span, subdiags }
            }
            ObligationCauseCode::IfExpressionWithNoElse => {
                ObligationCauseFailureCode::NoElse { span }
            }
            ObligationCauseCode::LetElse => {
                ObligationCauseFailureCode::NoDiverge { span, subdiags }
            }
            ObligationCauseCode::MainFunctionType => {
                ObligationCauseFailureCode::FnMainCorrectType { span }
            }
            &ObligationCauseCode::LangFunctionType(lang_item_name) => {
                ObligationCauseFailureCode::FnLangCorrectType { span, subdiags, lang_item_name }
            }
            ObligationCauseCode::IntrinsicType => {
                ObligationCauseFailureCode::IntrinsicCorrectType { span, subdiags }
            }
            ObligationCauseCode::MethodReceiver => {
                ObligationCauseFailureCode::MethodCorrectType { span, subdiags }
            }

            // In the case where we have no more specific thing to
            // say, also take a look at the error code, maybe we can
            // tailor to that.
            _ => match terr {
                TypeError::CyclicTy(ty)
                    if ty.is_closure() || ty.is_coroutine() || ty.is_coroutine_closure() =>
                {
                    ObligationCauseFailureCode::ClosureSelfref { span }
                }
                TypeError::ForceInlineCast => {
                    ObligationCauseFailureCode::CantCoerceForceInline { span, subdiags }
                }
                TypeError::IntrinsicCast => {
                    ObligationCauseFailureCode::CantCoerceIntrinsic { span, subdiags }
                }
                _ => ObligationCauseFailureCode::Generic { span, subdiags },
            },
        }
    }

    fn as_requirement_str(&self) -> &'static str {
        match self.code() {
            ObligationCauseCode::CompareImplItem { kind: ty::AssocKind::Fn { .. }, .. } => {
                "method type is compatible with trait"
            }
            ObligationCauseCode::CompareImplItem { kind: ty::AssocKind::Type { .. }, .. } => {
                "associated type is compatible with trait"
            }
            ObligationCauseCode::CompareImplItem { kind: ty::AssocKind::Const { .. }, .. } => {
                "const is compatible with trait"
            }
            ObligationCauseCode::MainFunctionType => "`main` function has the correct type",
            ObligationCauseCode::LangFunctionType(_) => "lang item function has the correct type",
            ObligationCauseCode::IntrinsicType => "intrinsic has the correct type",
            ObligationCauseCode::MethodReceiver => "method receiver has the correct type",
            _ => "types are compatible",
        }
    }
}

/// Newtype to allow implementing IntoDiagArg
pub struct ObligationCauseAsDiagArg<'tcx>(pub ObligationCause<'tcx>);

impl IntoDiagArg for ObligationCauseAsDiagArg<'_> {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> rustc_errors::DiagArgValue {
        let kind = match self.0.code() {
            ObligationCauseCode::CompareImplItem { kind: ty::AssocKind::Fn { .. }, .. } => {
                "method_compat"
            }
            ObligationCauseCode::CompareImplItem { kind: ty::AssocKind::Type { .. }, .. } => {
                "type_compat"
            }
            ObligationCauseCode::CompareImplItem { kind: ty::AssocKind::Const { .. }, .. } => {
                "const_compat"
            }
            ObligationCauseCode::MainFunctionType => "fn_main_correct_type",
            ObligationCauseCode::LangFunctionType(_) => "fn_lang_correct_type",
            ObligationCauseCode::IntrinsicType => "intrinsic_correct_type",
            ObligationCauseCode::MethodReceiver => "method_correct_type",
            _ => "other",
        }
        .into();
        rustc_errors::DiagArgValue::Str(kind)
    }
}

/// This is a bare signal of what kind of type we're dealing with. `ty::TyKind` tracks
/// extra information about each type, but we only care about the category.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum TyCategory {
    Closure,
    Opaque,
    OpaqueFuture,
    Coroutine(hir::CoroutineKind),
    Foreign,
}

impl fmt::Display for TyCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closure => "closure".fmt(f),
            Self::Opaque => "opaque type".fmt(f),
            Self::OpaqueFuture => "future".fmt(f),
            Self::Coroutine(gk) => gk.fmt(f),
            Self::Foreign => "foreign type".fmt(f),
        }
    }
}

impl TyCategory {
    pub fn from_ty(tcx: TyCtxt<'_>, ty: Ty<'_>) -> Option<(Self, DefId)> {
        match *ty.kind() {
            ty::Closure(def_id, _) => Some((Self::Closure, def_id)),
            ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }) => {
                let kind =
                    if tcx.ty_is_opaque_future(ty) { Self::OpaqueFuture } else { Self::Opaque };
                Some((kind, def_id))
            }
            ty::Coroutine(def_id, ..) => {
                Some((Self::Coroutine(tcx.coroutine_kind(def_id).unwrap()), def_id))
            }
            ty::Foreign(def_id) => Some((Self::Foreign, def_id)),
            _ => None,
        }
    }
}
