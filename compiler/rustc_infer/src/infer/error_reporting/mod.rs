//! Error Reporting Code for the inference engine
//!
//! Because of the way inference, and in particular region inference,
//! works, it often happens that errors are not detected until far after
//! the relevant line of code has been type-checked. Therefore, there is
//! an elaborate system to track why a particular constraint in the
//! inference graph arose so that we can explain to the user what gave
//! rise to a particular error.
//!
//! The basis of the system are the "origin" types. An "origin" is the
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

use super::lexical_region_resolve::RegionResolutionError;
use super::region_constraints::GenericKind;
use super::{InferCtxt, RegionVariableOrigin, SubregionOrigin, TypeTrace, ValuePairs};

use crate::infer;
use crate::traits::error_reporting::report_object_safety_error;
use crate::traits::{
    IfExpressionCause, MatchExpressionArmCause, ObligationCause, ObligationCauseCode,
    StatementAsExpression,
};

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{pluralize, struct_span_err};
use rustc_errors::{Applicability, DiagnosticBuilder, DiagnosticStyledString};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{Item, ItemKind, Node};
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::{
    self,
    subst::{Subst, SubstsRef},
    Region, Ty, TyCtxt, TypeFoldable,
};
use rustc_span::{sym, BytePos, DesugaringKind, Pos, Span};
use rustc_target::spec::abi;
use std::ops::ControlFlow;
use std::{cmp, fmt};

mod note;

mod need_type_info;
pub use need_type_info::TypeAnnotationNeeded;

pub mod nice_region_error;

pub(super) fn note_and_explain_region(
    tcx: TyCtxt<'tcx>,
    err: &mut DiagnosticBuilder<'_>,
    prefix: &str,
    region: ty::Region<'tcx>,
    suffix: &str,
) {
    let (description, span) = match *region {
        ty::ReEarlyBound(_) | ty::ReFree(_) | ty::ReStatic => {
            msg_span_from_free_region(tcx, region)
        }

        ty::ReEmpty(ty::UniverseIndex::ROOT) => ("the empty lifetime".to_owned(), None),

        // uh oh, hope no user ever sees THIS
        ty::ReEmpty(ui) => (format!("the empty lifetime in universe {:?}", ui), None),

        ty::RePlaceholder(_) => return,

        // FIXME(#13998) RePlaceholder should probably print like
        // ReFree rather than dumping Debug output on the user.
        //
        // We shouldn't really be having unification failures with ReVar
        // and ReLateBound though.
        ty::ReVar(_) | ty::ReLateBound(..) | ty::ReErased => {
            (format!("lifetime {:?}", region), None)
        }
    };

    emit_msg_span(err, prefix, description, span, suffix);
}

pub(super) fn note_and_explain_free_region(
    tcx: TyCtxt<'tcx>,
    err: &mut DiagnosticBuilder<'_>,
    prefix: &str,
    region: ty::Region<'tcx>,
    suffix: &str,
) {
    let (description, span) = msg_span_from_free_region(tcx, region);

    emit_msg_span(err, prefix, description, span, suffix);
}

fn msg_span_from_free_region(
    tcx: TyCtxt<'tcx>,
    region: ty::Region<'tcx>,
) -> (String, Option<Span>) {
    match *region {
        ty::ReEarlyBound(_) | ty::ReFree(_) => {
            msg_span_from_early_bound_and_free_regions(tcx, region)
        }
        ty::ReStatic => ("the static lifetime".to_owned(), None),
        ty::ReEmpty(ty::UniverseIndex::ROOT) => ("an empty lifetime".to_owned(), None),
        ty::ReEmpty(ui) => (format!("an empty lifetime in universe {:?}", ui), None),
        _ => bug!("{:?}", region),
    }
}

fn msg_span_from_early_bound_and_free_regions(
    tcx: TyCtxt<'tcx>,
    region: ty::Region<'tcx>,
) -> (String, Option<Span>) {
    let sm = tcx.sess.source_map();

    let scope = region.free_region_binding_scope(tcx);
    let node = tcx.hir().local_def_id_to_hir_id(scope.expect_local());
    let tag = match tcx.hir().find(node) {
        Some(Node::Block(_) | Node::Expr(_)) => "body",
        Some(Node::Item(it)) => item_scope_tag(&it),
        Some(Node::TraitItem(it)) => trait_item_scope_tag(&it),
        Some(Node::ImplItem(it)) => impl_item_scope_tag(&it),
        Some(Node::ForeignItem(it)) => foreign_item_scope_tag(&it),
        _ => unreachable!(),
    };
    let (prefix, span) = match *region {
        ty::ReEarlyBound(ref br) => {
            let mut sp = sm.guess_head_span(tcx.hir().span(node));
            if let Some(param) =
                tcx.hir().get_generics(scope).and_then(|generics| generics.get_named(br.name))
            {
                sp = param.span;
            }
            (format!("the lifetime `{}` as defined on", br.name), sp)
        }
        ty::ReFree(ty::FreeRegion {
            bound_region: ty::BoundRegionKind::BrNamed(_, name), ..
        }) => {
            let mut sp = sm.guess_head_span(tcx.hir().span(node));
            if let Some(param) =
                tcx.hir().get_generics(scope).and_then(|generics| generics.get_named(name))
            {
                sp = param.span;
            }
            (format!("the lifetime `{}` as defined on", name), sp)
        }
        ty::ReFree(ref fr) => match fr.bound_region {
            ty::BrAnon(idx) => {
                (format!("the anonymous lifetime #{} defined on", idx + 1), tcx.hir().span(node))
            }
            _ => (
                format!("the lifetime `{}` as defined on", region),
                sm.guess_head_span(tcx.hir().span(node)),
            ),
        },
        _ => bug!(),
    };
    let (msg, opt_span) = explain_span(tcx, tag, span);
    (format!("{} {}", prefix, msg), opt_span)
}

fn emit_msg_span(
    err: &mut DiagnosticBuilder<'_>,
    prefix: &str,
    description: String,
    span: Option<Span>,
    suffix: &str,
) {
    let message = format!("{}{}{}", prefix, description, suffix);

    if let Some(span) = span {
        err.span_note(span, &message);
    } else {
        err.note(&message);
    }
}

fn item_scope_tag(item: &hir::Item<'_>) -> &'static str {
    match item.kind {
        hir::ItemKind::Impl { .. } => "impl",
        hir::ItemKind::Struct(..) => "struct",
        hir::ItemKind::Union(..) => "union",
        hir::ItemKind::Enum(..) => "enum",
        hir::ItemKind::Trait(..) => "trait",
        hir::ItemKind::Fn(..) => "function body",
        _ => "item",
    }
}

fn trait_item_scope_tag(item: &hir::TraitItem<'_>) -> &'static str {
    match item.kind {
        hir::TraitItemKind::Fn(..) => "method body",
        hir::TraitItemKind::Const(..) | hir::TraitItemKind::Type(..) => "associated item",
    }
}

fn impl_item_scope_tag(item: &hir::ImplItem<'_>) -> &'static str {
    match item.kind {
        hir::ImplItemKind::Fn(..) => "method body",
        hir::ImplItemKind::Const(..) | hir::ImplItemKind::TyAlias(..) => "associated item",
    }
}

fn foreign_item_scope_tag(item: &hir::ForeignItem<'_>) -> &'static str {
    match item.kind {
        hir::ForeignItemKind::Fn(..) => "method body",
        hir::ForeignItemKind::Static(..) | hir::ForeignItemKind::Type => "associated item",
    }
}

fn explain_span(tcx: TyCtxt<'tcx>, heading: &str, span: Span) -> (String, Option<Span>) {
    let lo = tcx.sess.source_map().lookup_char_pos(span.lo());
    (format!("the {} at {}:{}", heading, lo.line, lo.col.to_usize() + 1), Some(span))
}

pub fn unexpected_hidden_region_diagnostic(
    tcx: TyCtxt<'tcx>,
    span: Span,
    hidden_ty: Ty<'tcx>,
    hidden_region: ty::Region<'tcx>,
) -> DiagnosticBuilder<'tcx> {
    let mut err = struct_span_err!(
        tcx.sess,
        span,
        E0700,
        "hidden type for `impl Trait` captures lifetime that does not appear in bounds",
    );

    // Explain the region we are capturing.
    match hidden_region {
        ty::ReEmpty(ty::UniverseIndex::ROOT) => {
            // All lifetimes shorter than the function body are `empty` in
            // lexical region resolution. The default explanation of "an empty
            // lifetime" isn't really accurate here.
            let message = format!(
                "hidden type `{}` captures lifetime smaller than the function body",
                hidden_ty
            );
            err.span_note(span, &message);
        }
        ty::ReEarlyBound(_) | ty::ReFree(_) | ty::ReStatic | ty::ReEmpty(_) => {
            // Assuming regionck succeeded (*), we ought to always be
            // capturing *some* region from the fn header, and hence it
            // ought to be free. So under normal circumstances, we will go
            // down this path which gives a decent human readable
            // explanation.
            //
            // (*) if not, the `tainted_by_errors` field would be set to
            // `Some(ErrorReported)` in any case, so we wouldn't be here at all.
            note_and_explain_free_region(
                tcx,
                &mut err,
                &format!("hidden type `{}` captures ", hidden_ty),
                hidden_region,
                "",
            );
        }
        _ => {
            // Ugh. This is a painful case: the hidden region is not one
            // that we can easily summarize or explain. This can happen
            // in a case like
            // `src/test/ui/multiple-lifetimes/ordinary-bounds-unsuited.rs`:
            //
            // ```
            // fn upper_bounds<'a, 'b>(a: Ordinary<'a>, b: Ordinary<'b>) -> impl Trait<'a, 'b> {
            //   if condition() { a } else { b }
            // }
            // ```
            //
            // Here the captured lifetime is the intersection of `'a` and
            // `'b`, which we can't quite express.

            // We can at least report a really cryptic error for now.
            note_and_explain_region(
                tcx,
                &mut err,
                &format!("hidden type `{}` captures ", hidden_ty),
                hidden_region,
                "",
            );
        }
    }

    err
}

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    pub fn report_region_errors(&self, errors: &Vec<RegionResolutionError<'tcx>>) {
        debug!("report_region_errors(): {} errors to start", errors.len());

        // try to pre-process the errors, which will group some of them
        // together into a `ProcessedErrors` group:
        let errors = self.process_errors(errors);

        debug!("report_region_errors: {} errors after preprocessing", errors.len());

        for error in errors {
            debug!("report_region_errors: error = {:?}", error);

            if !self.try_report_nice_region_error(&error) {
                match error.clone() {
                    // These errors could indicate all manner of different
                    // problems with many different solutions. Rather
                    // than generate a "one size fits all" error, what we
                    // attempt to do is go through a number of specific
                    // scenarios and try to find the best way to present
                    // the error. If all of these fails, we fall back to a rather
                    // general bit of code that displays the error information
                    RegionResolutionError::ConcreteFailure(origin, sub, sup) => {
                        if sub.is_placeholder() || sup.is_placeholder() {
                            self.report_placeholder_failure(origin, sub, sup).emit();
                        } else {
                            self.report_concrete_failure(origin, sub, sup).emit();
                        }
                    }

                    RegionResolutionError::GenericBoundFailure(origin, param_ty, sub) => {
                        self.report_generic_bound_failure(
                            origin.span(),
                            Some(origin),
                            param_ty,
                            sub,
                        );
                    }

                    RegionResolutionError::SubSupConflict(
                        _,
                        var_origin,
                        sub_origin,
                        sub_r,
                        sup_origin,
                        sup_r,
                    ) => {
                        if sub_r.is_placeholder() {
                            self.report_placeholder_failure(sub_origin, sub_r, sup_r).emit();
                        } else if sup_r.is_placeholder() {
                            self.report_placeholder_failure(sup_origin, sub_r, sup_r).emit();
                        } else {
                            self.report_sub_sup_conflict(
                                var_origin, sub_origin, sub_r, sup_origin, sup_r,
                            );
                        }
                    }

                    RegionResolutionError::UpperBoundUniverseConflict(
                        _,
                        _,
                        var_universe,
                        sup_origin,
                        sup_r,
                    ) => {
                        assert!(sup_r.is_placeholder());

                        // Make a dummy value for the "sub region" --
                        // this is the initial value of the
                        // placeholder. In practice, we expect more
                        // tailored errors that don't really use this
                        // value.
                        let sub_r = self.tcx.mk_region(ty::ReEmpty(var_universe));

                        self.report_placeholder_failure(sup_origin, sub_r, sup_r).emit();
                    }

                    RegionResolutionError::MemberConstraintFailure {
                        hidden_ty,
                        member_region,
                        span,
                    } => {
                        let hidden_ty = self.resolve_vars_if_possible(hidden_ty);
                        unexpected_hidden_region_diagnostic(
                            self.tcx,
                            span,
                            hidden_ty,
                            member_region,
                        )
                        .emit();
                    }
                }
            }
        }
    }

    // This method goes through all the errors and try to group certain types
    // of error together, for the purpose of suggesting explicit lifetime
    // parameters to the user. This is done so that we can have a more
    // complete view of what lifetimes should be the same.
    // If the return value is an empty vector, it means that processing
    // failed (so the return value of this method should not be used).
    //
    // The method also attempts to weed out messages that seem like
    // duplicates that will be unhelpful to the end-user. But
    // obviously it never weeds out ALL errors.
    fn process_errors(
        &self,
        errors: &[RegionResolutionError<'tcx>],
    ) -> Vec<RegionResolutionError<'tcx>> {
        debug!("process_errors()");

        // We want to avoid reporting generic-bound failures if we can
        // avoid it: these have a very high rate of being unhelpful in
        // practice. This is because they are basically secondary
        // checks that test the state of the region graph after the
        // rest of inference is done, and the other kinds of errors
        // indicate that the region constraint graph is internally
        // inconsistent, so these test results are likely to be
        // meaningless.
        //
        // Therefore, we filter them out of the list unless they are
        // the only thing in the list.

        let is_bound_failure = |e: &RegionResolutionError<'tcx>| match *e {
            RegionResolutionError::GenericBoundFailure(..) => true,
            RegionResolutionError::ConcreteFailure(..)
            | RegionResolutionError::SubSupConflict(..)
            | RegionResolutionError::UpperBoundUniverseConflict(..)
            | RegionResolutionError::MemberConstraintFailure { .. } => false,
        };

        let mut errors = if errors.iter().all(|e| is_bound_failure(e)) {
            errors.to_owned()
        } else {
            errors.iter().filter(|&e| !is_bound_failure(e)).cloned().collect()
        };

        // sort the errors by span, for better error message stability.
        errors.sort_by_key(|u| match *u {
            RegionResolutionError::ConcreteFailure(ref sro, _, _) => sro.span(),
            RegionResolutionError::GenericBoundFailure(ref sro, _, _) => sro.span(),
            RegionResolutionError::SubSupConflict(_, ref rvo, _, _, _, _) => rvo.span(),
            RegionResolutionError::UpperBoundUniverseConflict(_, ref rvo, _, _, _) => rvo.span(),
            RegionResolutionError::MemberConstraintFailure { span, .. } => span,
        });
        errors
    }

    /// Adds a note if the types come from similarly named crates
    fn check_and_note_conflicting_crates(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        terr: &TypeError<'tcx>,
    ) {
        use hir::def_id::CrateNum;
        use rustc_hir::definitions::DisambiguatedDefPathData;
        use ty::print::Printer;
        use ty::subst::GenericArg;

        struct AbsolutePathPrinter<'tcx> {
            tcx: TyCtxt<'tcx>,
        }

        struct NonTrivialPath;

        impl<'tcx> Printer<'tcx> for AbsolutePathPrinter<'tcx> {
            type Error = NonTrivialPath;

            type Path = Vec<String>;
            type Region = !;
            type Type = !;
            type DynExistential = !;
            type Const = !;

            fn tcx<'a>(&'a self) -> TyCtxt<'tcx> {
                self.tcx
            }

            fn print_region(self, _region: ty::Region<'_>) -> Result<Self::Region, Self::Error> {
                Err(NonTrivialPath)
            }

            fn print_type(self, _ty: Ty<'tcx>) -> Result<Self::Type, Self::Error> {
                Err(NonTrivialPath)
            }

            fn print_dyn_existential(
                self,
                _predicates: &'tcx ty::List<ty::Binder<ty::ExistentialPredicate<'tcx>>>,
            ) -> Result<Self::DynExistential, Self::Error> {
                Err(NonTrivialPath)
            }

            fn print_const(self, _ct: &'tcx ty::Const<'tcx>) -> Result<Self::Const, Self::Error> {
                Err(NonTrivialPath)
            }

            fn path_crate(self, cnum: CrateNum) -> Result<Self::Path, Self::Error> {
                Ok(vec![self.tcx.original_crate_name(cnum).to_string()])
            }
            fn path_qualified(
                self,
                _self_ty: Ty<'tcx>,
                _trait_ref: Option<ty::TraitRef<'tcx>>,
            ) -> Result<Self::Path, Self::Error> {
                Err(NonTrivialPath)
            }

            fn path_append_impl(
                self,
                _print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
                _disambiguated_data: &DisambiguatedDefPathData,
                _self_ty: Ty<'tcx>,
                _trait_ref: Option<ty::TraitRef<'tcx>>,
            ) -> Result<Self::Path, Self::Error> {
                Err(NonTrivialPath)
            }
            fn path_append(
                self,
                print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
                disambiguated_data: &DisambiguatedDefPathData,
            ) -> Result<Self::Path, Self::Error> {
                let mut path = print_prefix(self)?;
                path.push(disambiguated_data.to_string());
                Ok(path)
            }
            fn path_generic_args(
                self,
                print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
                _args: &[GenericArg<'tcx>],
            ) -> Result<Self::Path, Self::Error> {
                print_prefix(self)
            }
        }

        let report_path_match = |err: &mut DiagnosticBuilder<'_>, did1: DefId, did2: DefId| {
            // Only external crates, if either is from a local
            // module we could have false positives
            if !(did1.is_local() || did2.is_local()) && did1.krate != did2.krate {
                let abs_path =
                    |def_id| AbsolutePathPrinter { tcx: self.tcx }.print_def_path(def_id, &[]);

                // We compare strings because DefPath can be different
                // for imported and non-imported crates
                let same_path = || -> Result<_, NonTrivialPath> {
                    Ok(self.tcx.def_path_str(did1) == self.tcx.def_path_str(did2)
                        || abs_path(did1)? == abs_path(did2)?)
                };
                if same_path().unwrap_or(false) {
                    let crate_name = self.tcx.crate_name(did1.krate);
                    err.note(&format!(
                        "perhaps two different versions of crate `{}` are being used?",
                        crate_name
                    ));
                }
            }
        };
        match *terr {
            TypeError::Sorts(ref exp_found) => {
                // if they are both "path types", there's a chance of ambiguity
                // due to different versions of the same crate
                if let (&ty::Adt(exp_adt, _), &ty::Adt(found_adt, _)) =
                    (exp_found.expected.kind(), exp_found.found.kind())
                {
                    report_path_match(err, exp_adt.did, found_adt.did);
                }
            }
            TypeError::Traits(ref exp_found) => {
                report_path_match(err, exp_found.expected, exp_found.found);
            }
            _ => (), // FIXME(#22750) handle traits and stuff
        }
    }

    fn note_error_origin(
        &self,
        err: &mut DiagnosticBuilder<'tcx>,
        cause: &ObligationCause<'tcx>,
        exp_found: Option<ty::error::ExpectedFound<Ty<'tcx>>>,
    ) {
        match cause.code {
            ObligationCauseCode::Pattern { origin_expr: true, span: Some(span), root_ty } => {
                let ty = self.resolve_vars_if_possible(root_ty);
                if ty.is_suggestable() {
                    // don't show type `_`
                    err.span_label(span, format!("this expression has type `{}`", ty));
                }
                if let Some(ty::error::ExpectedFound { found, .. }) = exp_found {
                    if ty.is_box() && ty.boxed_ty() == found {
                        if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
                            err.span_suggestion(
                                span,
                                "consider dereferencing the boxed value",
                                format!("*{}", snippet),
                                Applicability::MachineApplicable,
                            );
                        }
                    }
                }
            }
            ObligationCauseCode::Pattern { origin_expr: false, span: Some(span), .. } => {
                err.span_label(span, "expected due to this");
            }
            ObligationCauseCode::MatchExpressionArm(box MatchExpressionArmCause {
                semi_span,
                source,
                ref prior_arms,
                last_ty,
                scrut_hir_id,
                opt_suggest_box_span,
                arm_span,
                scrut_span,
                ..
            }) => match source {
                hir::MatchSource::IfLetDesugar { .. } => {
                    let msg = "`if let` arms have incompatible types";
                    err.span_label(cause.span, msg);
                    if let Some(ret_sp) = opt_suggest_box_span {
                        self.suggest_boxing_for_return_impl_trait(
                            err,
                            ret_sp,
                            prior_arms.iter().chain(std::iter::once(&arm_span)).map(|s| *s),
                        );
                    }
                }
                hir::MatchSource::TryDesugar => {
                    if let Some(ty::error::ExpectedFound { expected, .. }) = exp_found {
                        let scrut_expr = self.tcx.hir().expect_expr(scrut_hir_id);
                        let scrut_ty = if let hir::ExprKind::Call(_, args) = &scrut_expr.kind {
                            let arg_expr = args.first().expect("try desugaring call w/out arg");
                            self.in_progress_typeck_results.and_then(|typeck_results| {
                                typeck_results.borrow().expr_ty_opt(arg_expr)
                            })
                        } else {
                            bug!("try desugaring w/out call expr as scrutinee");
                        };

                        match scrut_ty {
                            Some(ty) if expected == ty => {
                                let source_map = self.tcx.sess.source_map();
                                err.span_suggestion(
                                    source_map.end_point(cause.span),
                                    "try removing this `?`",
                                    "".to_string(),
                                    Applicability::MachineApplicable,
                                );
                            }
                            _ => {}
                        }
                    }
                }
                _ => {
                    // `last_ty` can be `!`, `expected` will have better info when present.
                    let t = self.resolve_vars_if_possible(match exp_found {
                        Some(ty::error::ExpectedFound { expected, .. }) => expected,
                        _ => last_ty,
                    });
                    let source_map = self.tcx.sess.source_map();
                    let mut any_multiline_arm = source_map.is_multiline(arm_span);
                    if prior_arms.len() <= 4 {
                        for sp in prior_arms {
                            any_multiline_arm |= source_map.is_multiline(*sp);
                            err.span_label(*sp, format!("this is found to be of type `{}`", t));
                        }
                    } else if let Some(sp) = prior_arms.last() {
                        any_multiline_arm |= source_map.is_multiline(*sp);
                        err.span_label(
                            *sp,
                            format!("this and all prior arms are found to be of type `{}`", t),
                        );
                    }
                    let outer_error_span = if any_multiline_arm {
                        // Cover just `match` and the scrutinee expression, not
                        // the entire match body, to reduce diagram noise.
                        cause.span.shrink_to_lo().to(scrut_span)
                    } else {
                        cause.span
                    };
                    let msg = "`match` arms have incompatible types";
                    err.span_label(outer_error_span, msg);
                    if let Some((sp, boxed)) = semi_span {
                        if let (StatementAsExpression::NeedsBoxing, [.., prior_arm]) =
                            (boxed, &prior_arms[..])
                        {
                            err.multipart_suggestion(
                                "consider removing this semicolon and boxing the expressions",
                                vec![
                                    (prior_arm.shrink_to_lo(), "Box::new(".to_string()),
                                    (prior_arm.shrink_to_hi(), ")".to_string()),
                                    (arm_span.shrink_to_lo(), "Box::new(".to_string()),
                                    (arm_span.shrink_to_hi(), ")".to_string()),
                                    (sp, String::new()),
                                ],
                                Applicability::HasPlaceholders,
                            );
                        } else if matches!(boxed, StatementAsExpression::NeedsBoxing) {
                            err.span_suggestion_short(
                                sp,
                                "consider removing this semicolon and boxing the expressions",
                                String::new(),
                                Applicability::MachineApplicable,
                            );
                        } else {
                            err.span_suggestion_short(
                                sp,
                                "consider removing this semicolon",
                                String::new(),
                                Applicability::MachineApplicable,
                            );
                        }
                    }
                    if let Some(ret_sp) = opt_suggest_box_span {
                        // Get return type span and point to it.
                        self.suggest_boxing_for_return_impl_trait(
                            err,
                            ret_sp,
                            prior_arms.iter().chain(std::iter::once(&arm_span)).map(|s| *s),
                        );
                    }
                }
            },
            ObligationCauseCode::IfExpression(box IfExpressionCause {
                then,
                else_sp,
                outer,
                semicolon,
                opt_suggest_box_span,
            }) => {
                err.span_label(then, "expected because of this");
                if let Some(sp) = outer {
                    err.span_label(sp, "`if` and `else` have incompatible types");
                }
                if let Some((sp, boxed)) = semicolon {
                    if matches!(boxed, StatementAsExpression::NeedsBoxing) {
                        err.multipart_suggestion(
                            "consider removing this semicolon and boxing the expression",
                            vec![
                                (then.shrink_to_lo(), "Box::new(".to_string()),
                                (then.shrink_to_hi(), ")".to_string()),
                                (else_sp.shrink_to_lo(), "Box::new(".to_string()),
                                (else_sp.shrink_to_hi(), ")".to_string()),
                                (sp, String::new()),
                            ],
                            Applicability::MachineApplicable,
                        );
                    } else {
                        err.span_suggestion_short(
                            sp,
                            "consider removing this semicolon",
                            String::new(),
                            Applicability::MachineApplicable,
                        );
                    }
                }
                if let Some(ret_sp) = opt_suggest_box_span {
                    self.suggest_boxing_for_return_impl_trait(
                        err,
                        ret_sp,
                        vec![then, else_sp].into_iter(),
                    );
                }
            }
            _ => (),
        }
    }

    fn suggest_boxing_for_return_impl_trait(
        &self,
        err: &mut DiagnosticBuilder<'tcx>,
        return_sp: Span,
        arm_spans: impl Iterator<Item = Span>,
    ) {
        err.multipart_suggestion(
            "you could change the return type to be a boxed trait object",
            vec![
                (return_sp.with_hi(return_sp.lo() + BytePos(4)), "Box<dyn".to_string()),
                (return_sp.shrink_to_hi(), ">".to_string()),
            ],
            Applicability::MaybeIncorrect,
        );
        let sugg = arm_spans
            .flat_map(|sp| {
                vec![
                    (sp.shrink_to_lo(), "Box::new(".to_string()),
                    (sp.shrink_to_hi(), ")".to_string()),
                ]
                .into_iter()
            })
            .collect::<Vec<_>>();
        err.multipart_suggestion(
            "if you change the return type to expect trait objects, box the returned expressions",
            sugg,
            Applicability::MaybeIncorrect,
        );
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
        value: &mut DiagnosticStyledString,
        other_value: &mut DiagnosticStyledString,
        name: String,
        sub: ty::subst::SubstsRef<'tcx>,
        pos: usize,
        other_ty: Ty<'tcx>,
    ) {
        // `value` and `other_value` hold two incomplete type representation for display.
        // `name` is the path of both types being compared. `sub`
        value.push_highlighted(name);
        let len = sub.len();
        if len > 0 {
            value.push_highlighted("<");
        }

        // Output the lifetimes for the first type
        let lifetimes = sub
            .regions()
            .map(|lifetime| {
                let s = lifetime.to_string();
                if s.is_empty() { "'_".to_string() } else { s }
            })
            .collect::<Vec<_>>()
            .join(", ");
        if !lifetimes.is_empty() {
            if sub.regions().count() < len {
                value.push_normal(lifetimes + ", ");
            } else {
                value.push_normal(lifetimes);
            }
        }

        // Highlight all the type arguments that aren't at `pos` and compare the type argument at
        // `pos` and `other_ty`.
        for (i, type_arg) in sub.types().enumerate() {
            if i == pos {
                let values = self.cmp(type_arg, other_ty);
                value.0.extend((values.0).0);
                other_value.0.extend((values.1).0);
            } else {
                value.push_highlighted(type_arg.to_string());
            }

            if len > 0 && i != len - 1 {
                value.push_normal(", ");
            }
        }
        if len > 0 {
            value.push_highlighted(">");
        }
    }

    /// If `other_ty` is the same as a type argument present in `sub`, highlight `path` in `t1_out`,
    /// as that is the difference to the other type.
    ///
    /// For the following code:
    ///
    /// ```no_run
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
        mut t1_out: &mut DiagnosticStyledString,
        mut t2_out: &mut DiagnosticStyledString,
        path: String,
        sub: ty::subst::SubstsRef<'tcx>,
        other_path: String,
        other_ty: Ty<'tcx>,
    ) -> Option<()> {
        for (i, ta) in sub.types().enumerate() {
            if ta == other_ty {
                self.highlight_outer(&mut t1_out, &mut t2_out, path, sub, i, &other_ty);
                return Some(());
            }
            if let ty::Adt(def, _) = ta.kind() {
                let path_ = self.tcx.def_path_str(def.did);
                if path_ == other_path {
                    self.highlight_outer(&mut t1_out, &mut t2_out, path, sub, i, &other_ty);
                    return Some(());
                }
            }
        }
        None
    }

    /// Adds a `,` to the type representation only if it is appropriate.
    fn push_comma(
        &self,
        value: &mut DiagnosticStyledString,
        other_value: &mut DiagnosticStyledString,
        len: usize,
        pos: usize,
    ) {
        if len > 0 && pos != len - 1 {
            value.push_normal(", ");
            other_value.push_normal(", ");
        }
    }

    /// For generic types with parameters with defaults, remove the parameters corresponding to
    /// the defaults. This repeats a lot of the logic found in `ty::print::pretty`.
    fn strip_generic_default_params(
        &self,
        def_id: DefId,
        substs: ty::subst::SubstsRef<'tcx>,
    ) -> SubstsRef<'tcx> {
        let generics = self.tcx.generics_of(def_id);
        let mut num_supplied_defaults = 0;
        let mut type_params = generics
            .params
            .iter()
            .rev()
            .filter_map(|param| match param.kind {
                ty::GenericParamDefKind::Lifetime => None,
                ty::GenericParamDefKind::Type { has_default, .. } => {
                    Some((param.def_id, has_default))
                }
                ty::GenericParamDefKind::Const => None, // FIXME(const_generics_defaults)
            })
            .peekable();
        let has_default = {
            let has_default = type_params.peek().map(|(_, has_default)| has_default);
            *has_default.unwrap_or(&false)
        };
        if has_default {
            let types = substs.types().rev();
            for ((def_id, has_default), actual) in type_params.zip(types) {
                if !has_default {
                    break;
                }
                if self.tcx.type_of(def_id).subst(self.tcx, substs) != actual {
                    break;
                }
                num_supplied_defaults += 1;
            }
        }
        let len = generics.params.len();
        let mut generics = generics.clone();
        generics.params.truncate(len - num_supplied_defaults);
        substs.truncate_to(self.tcx, &generics)
    }

    /// Given two `fn` signatures highlight only sub-parts that are different.
    fn cmp_fn_sig(
        &self,
        sig1: &ty::PolyFnSig<'tcx>,
        sig2: &ty::PolyFnSig<'tcx>,
    ) -> (DiagnosticStyledString, DiagnosticStyledString) {
        let get_lifetimes = |sig| {
            use rustc_hir::def::Namespace;
            let mut s = String::new();
            let (_, (sig, reg)) = ty::print::FmtPrinter::new(self.tcx, &mut s, Namespace::TypeNS)
                .name_all_regions(sig)
                .unwrap();
            let lts: Vec<String> = reg.into_iter().map(|(_, kind)| kind.to_string()).collect();
            (if lts.is_empty() { String::new() } else { format!("for<{}> ", lts.join(", ")) }, sig)
        };

        let (lt1, sig1) = get_lifetimes(sig1);
        let (lt2, sig2) = get_lifetimes(sig2);

        // unsafe extern "C" for<'a> fn(&'a T) -> &'a T
        let mut values = (
            DiagnosticStyledString::normal("".to_string()),
            DiagnosticStyledString::normal("".to_string()),
        );

        // unsafe extern "C" for<'a> fn(&'a T) -> &'a T
        // ^^^^^^
        values.0.push(sig1.unsafety.prefix_str(), sig1.unsafety != sig2.unsafety);
        values.1.push(sig2.unsafety.prefix_str(), sig1.unsafety != sig2.unsafety);

        // unsafe extern "C" for<'a> fn(&'a T) -> &'a T
        //        ^^^^^^^^^^
        if sig1.abi != abi::Abi::Rust {
            values.0.push(format!("extern {} ", sig1.abi), sig1.abi != sig2.abi);
        }
        if sig2.abi != abi::Abi::Rust {
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
            for (i, (l, r)) in sig1.inputs().iter().zip(sig2.inputs().iter()).enumerate() {
                let (x1, x2) = self.cmp(l, r);
                (values.0).0.extend(x1.0);
                (values.1).0.extend(x2.0);
                self.push_comma(&mut values.0, &mut values.1, len1, i);
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
        if !output1.is_unit() {
            values.0.push_normal(" -> ");
            (values.0).0.extend(x1.0);
        }
        if !output2.is_unit() {
            values.1.push_normal(" -> ");
            (values.1).0.extend(x2.0);
        }
        values
    }

    /// Compares two given types, eliding parts that are the same between them and highlighting
    /// relevant differences, and return two representation of those types for highlighted printing.
    fn cmp(&self, t1: Ty<'tcx>, t2: Ty<'tcx>) -> (DiagnosticStyledString, DiagnosticStyledString) {
        debug!("cmp(t1={}, t1.kind={:?}, t2={}, t2.kind={:?})", t1, t1.kind(), t2, t2.kind());

        // helper functions
        fn equals<'tcx>(a: Ty<'tcx>, b: Ty<'tcx>) -> bool {
            match (a.kind(), b.kind()) {
                (a, b) if *a == *b => true,
                (&ty::Int(_), &ty::Infer(ty::InferTy::IntVar(_)))
                | (
                    &ty::Infer(ty::InferTy::IntVar(_)),
                    &ty::Int(_) | &ty::Infer(ty::InferTy::IntVar(_)),
                )
                | (&ty::Float(_), &ty::Infer(ty::InferTy::FloatVar(_)))
                | (
                    &ty::Infer(ty::InferTy::FloatVar(_)),
                    &ty::Float(_) | &ty::Infer(ty::InferTy::FloatVar(_)),
                ) => true,
                _ => false,
            }
        }

        fn push_ty_ref<'tcx>(
            region: &ty::Region<'tcx>,
            ty: Ty<'tcx>,
            mutbl: hir::Mutability,
            s: &mut DiagnosticStyledString,
        ) {
            let mut r = region.to_string();
            if r == "'_" {
                r.clear();
            } else {
                r.push(' ');
            }
            s.push_highlighted(format!("&{}{}", r, mutbl.prefix_str()));
            s.push_normal(ty.to_string());
        }

        // process starts here
        match (t1.kind(), t2.kind()) {
            (&ty::Adt(def1, sub1), &ty::Adt(def2, sub2)) => {
                let sub_no_defaults_1 = self.strip_generic_default_params(def1.did, sub1);
                let sub_no_defaults_2 = self.strip_generic_default_params(def2.did, sub2);
                let mut values = (DiagnosticStyledString::new(), DiagnosticStyledString::new());
                let path1 = self.tcx.def_path_str(def1.did);
                let path2 = self.tcx.def_path_str(def2.did);
                if def1.did == def2.did {
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
                    let remainder1: Vec<_> = sub1.types().skip(common_len).collect();
                    let remainder2: Vec<_> = sub2.types().skip(common_len).collect();
                    let common_default_params = remainder1
                        .iter()
                        .rev()
                        .zip(remainder2.iter().rev())
                        .filter(|(a, b)| a == b)
                        .count();
                    let len = sub1.len() - common_default_params;
                    let consts_offset = len - sub1.consts().count();

                    // Only draw `<...>` if there're lifetime/type arguments.
                    if len > 0 {
                        values.0.push_normal("<");
                        values.1.push_normal("<");
                    }

                    fn lifetime_display(lifetime: Region<'_>) -> String {
                        let s = lifetime.to_string();
                        if s.is_empty() { "'_".to_string() } else { s }
                    }
                    // At one point we'd like to elide all lifetimes here, they are irrelevant for
                    // all diagnostics that use this output
                    //
                    //     Foo<'x, '_, Bar>
                    //     Foo<'y, '_, Qux>
                    //         ^^  ^^  --- type arguments are not elided
                    //         |   |
                    //         |   elided as they were the same
                    //         not elided, they were different, but irrelevant
                    let lifetimes = sub1.regions().zip(sub2.regions());
                    for (i, lifetimes) in lifetimes.enumerate() {
                        let l1 = lifetime_display(lifetimes.0);
                        let l2 = lifetime_display(lifetimes.1);
                        if lifetimes.0 == lifetimes.1 {
                            values.0.push_normal("'_");
                            values.1.push_normal("'_");
                        } else {
                            values.0.push_highlighted(l1);
                            values.1.push_highlighted(l2);
                        }
                        self.push_comma(&mut values.0, &mut values.1, len, i);
                    }

                    // We're comparing two types with the same path, so we compare the type
                    // arguments for both. If they are the same, do not highlight and elide from the
                    // output.
                    //     Foo<_, Bar>
                    //     Foo<_, Qux>
                    //         ^ elided type as this type argument was the same in both sides
                    let type_arguments = sub1.types().zip(sub2.types());
                    let regions_len = sub1.regions().count();
                    let num_display_types = consts_offset - regions_len;
                    for (i, (ta1, ta2)) in type_arguments.take(num_display_types).enumerate() {
                        let i = i + regions_len;
                        if ta1 == ta2 {
                            values.0.push_normal("_");
                            values.1.push_normal("_");
                        } else {
                            let (x1, x2) = self.cmp(ta1, ta2);
                            (values.0).0.extend(x1.0);
                            (values.1).0.extend(x2.0);
                        }
                        self.push_comma(&mut values.0, &mut values.1, len, i);
                    }

                    // Do the same for const arguments, if they are equal, do not highlight and
                    // elide them from the output.
                    let const_arguments = sub1.consts().zip(sub2.consts());
                    for (i, (ca1, ca2)) in const_arguments.enumerate() {
                        let i = i + consts_offset;
                        if ca1 == ca2 {
                            values.0.push_normal("_");
                            values.1.push_normal("_");
                        } else {
                            values.0.push_highlighted(ca1.to_string());
                            values.1.push_highlighted(ca2.to_string());
                        }
                        self.push_comma(&mut values.0, &mut values.1, len, i);
                    }

                    // Close the type argument bracket.
                    // Only draw `<...>` if there're lifetime/type arguments.
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
                    if self
                        .cmp_type_arg(
                            &mut values.0,
                            &mut values.1,
                            path1.clone(),
                            sub_no_defaults_1,
                            path2.clone(),
                            &t2,
                        )
                        .is_some()
                    {
                        return values;
                    }
                    // Check for case:
                    //     let x: Bar<Qux> = y:<Foo<Bar<Qux>>>();
                    //     Bar<Qux>
                    //     Foo<Bar<Qux>>
                    //         ------- this type argument is exactly the same as the other type
                    if self
                        .cmp_type_arg(
                            &mut values.1,
                            &mut values.0,
                            path2,
                            sub_no_defaults_2,
                            path1,
                            &t1,
                        )
                        .is_some()
                    {
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
                    let split_idx: usize = t1_str
                        .split(SEPARATOR)
                        .zip(t2_str.split(SEPARATOR))
                        .take_while(|(mod1_str, mod2_str)| mod1_str == mod2_str)
                        .map(|(mod_str, _)| mod_str.len() + separator_len)
                        .sum();

                    debug!(
                        "cmp: separator_len={}, split_idx={}, min_len={}",
                        separator_len, split_idx, min_len
                    );

                    if split_idx >= min_len {
                        // paths are identical, highlight everything
                        (
                            DiagnosticStyledString::highlighted(t1_str),
                            DiagnosticStyledString::highlighted(t2_str),
                        )
                    } else {
                        let (common, uniq1) = t1_str.split_at(split_idx);
                        let (_, uniq2) = t2_str.split_at(split_idx);
                        debug!("cmp: common={}, uniq1={}, uniq2={}", common, uniq1, uniq2);

                        values.0.push_normal(common);
                        values.0.push_highlighted(uniq1);
                        values.1.push_normal(common);
                        values.1.push_highlighted(uniq2);

                        values
                    }
                }
            }

            // When finding T != &T, highlight only the borrow
            (&ty::Ref(r1, ref_ty1, mutbl1), _) if equals(&ref_ty1, &t2) => {
                let mut values = (DiagnosticStyledString::new(), DiagnosticStyledString::new());
                push_ty_ref(&r1, ref_ty1, mutbl1, &mut values.0);
                values.1.push_normal(t2.to_string());
                values
            }
            (_, &ty::Ref(r2, ref_ty2, mutbl2)) if equals(&t1, &ref_ty2) => {
                let mut values = (DiagnosticStyledString::new(), DiagnosticStyledString::new());
                values.0.push_normal(t1.to_string());
                push_ty_ref(&r2, ref_ty2, mutbl2, &mut values.1);
                values
            }

            // When encountering &T != &mut T, highlight only the borrow
            (&ty::Ref(r1, ref_ty1, mutbl1), &ty::Ref(r2, ref_ty2, mutbl2))
                if equals(&ref_ty1, &ref_ty2) =>
            {
                let mut values = (DiagnosticStyledString::new(), DiagnosticStyledString::new());
                push_ty_ref(&r1, ref_ty1, mutbl1, &mut values.0);
                push_ty_ref(&r2, ref_ty2, mutbl2, &mut values.1);
                values
            }

            // When encountering tuples of the same size, highlight only the differing types
            (&ty::Tuple(substs1), &ty::Tuple(substs2)) if substs1.len() == substs2.len() => {
                let mut values =
                    (DiagnosticStyledString::normal("("), DiagnosticStyledString::normal("("));
                let len = substs1.len();
                for (i, (left, right)) in substs1.types().zip(substs2.types()).enumerate() {
                    let (x1, x2) = self.cmp(left, right);
                    (values.0).0.extend(x1.0);
                    (values.1).0.extend(x2.0);
                    self.push_comma(&mut values.0, &mut values.1, len, i);
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

            (ty::FnDef(did1, substs1), ty::FnDef(did2, substs2)) => {
                let sig1 = self.tcx.fn_sig(*did1).subst(self.tcx, substs1);
                let sig2 = self.tcx.fn_sig(*did2).subst(self.tcx, substs2);
                let mut values = self.cmp_fn_sig(&sig1, &sig2);
                let path1 = format!(" {{{}}}", self.tcx.def_path_str_with_substs(*did1, substs1));
                let path2 = format!(" {{{}}}", self.tcx.def_path_str_with_substs(*did2, substs2));
                let same_path = path1 == path2;
                values.0.push(path1, !same_path);
                values.1.push(path2, !same_path);
                values
            }

            (ty::FnDef(did1, substs1), ty::FnPtr(sig2)) => {
                let sig1 = self.tcx.fn_sig(*did1).subst(self.tcx, substs1);
                let mut values = self.cmp_fn_sig(&sig1, sig2);
                values.0.push_highlighted(format!(
                    " {{{}}}",
                    self.tcx.def_path_str_with_substs(*did1, substs1)
                ));
                values
            }

            (ty::FnPtr(sig1), ty::FnDef(did2, substs2)) => {
                let sig2 = self.tcx.fn_sig(*did2).subst(self.tcx, substs2);
                let mut values = self.cmp_fn_sig(sig1, &sig2);
                values.1.push_normal(format!(
                    " {{{}}}",
                    self.tcx.def_path_str_with_substs(*did2, substs2)
                ));
                values
            }

            (ty::FnPtr(sig1), ty::FnPtr(sig2)) => self.cmp_fn_sig(sig1, sig2),

            _ => {
                if t1 == t2 {
                    // The two types are the same, elide and don't highlight.
                    (DiagnosticStyledString::normal("_"), DiagnosticStyledString::normal("_"))
                } else {
                    // We couldn't find anything in common, highlight everything.
                    (
                        DiagnosticStyledString::highlighted(t1.to_string()),
                        DiagnosticStyledString::highlighted(t2.to_string()),
                    )
                }
            }
        }
    }

    pub fn note_type_err(
        &self,
        diag: &mut DiagnosticBuilder<'tcx>,
        cause: &ObligationCause<'tcx>,
        secondary_span: Option<(Span, String)>,
        mut values: Option<ValuePairs<'tcx>>,
        terr: &TypeError<'tcx>,
    ) {
        let span = cause.span(self.tcx);
        debug!("note_type_err cause={:?} values={:?}, terr={:?}", cause, values, terr);

        // For some types of errors, expected-found does not make
        // sense, so just ignore the values we were given.
        if let TypeError::CyclicTy(_) = terr {
            values = None;
        }
        struct OpaqueTypesVisitor<'tcx> {
            types: FxHashMap<TyCategory, FxHashSet<Span>>,
            expected: FxHashMap<TyCategory, FxHashSet<Span>>,
            found: FxHashMap<TyCategory, FxHashSet<Span>>,
            ignore_span: Span,
            tcx: TyCtxt<'tcx>,
        }

        impl<'tcx> OpaqueTypesVisitor<'tcx> {
            fn visit_expected_found(
                tcx: TyCtxt<'tcx>,
                expected: Ty<'tcx>,
                found: Ty<'tcx>,
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

            fn report(&self, err: &mut DiagnosticBuilder<'_>) {
                self.add_labels_for_types(err, "expected", &self.expected);
                self.add_labels_for_types(err, "found", &self.found);
            }

            fn add_labels_for_types(
                &self,
                err: &mut DiagnosticBuilder<'_>,
                target: &str,
                types: &FxHashMap<TyCategory, FxHashSet<Span>>,
            ) {
                for (key, values) in types.iter() {
                    let count = values.len();
                    let kind = key.descr();
                    for sp in values {
                        err.span_label(
                            *sp,
                            format!(
                                "{}{}{} {}{}",
                                if sp.is_desugaring(DesugaringKind::Async) {
                                    "the `Output` of this `async fn`'s "
                                } else if count == 1 {
                                    "the "
                                } else {
                                    ""
                                },
                                if count > 1 { "one of the " } else { "" },
                                target,
                                kind,
                                pluralize!(count),
                            ),
                        );
                    }
                }
            }
        }

        impl<'tcx> ty::fold::TypeVisitor<'tcx> for OpaqueTypesVisitor<'tcx> {
            fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
                if let Some((kind, def_id)) = TyCategory::from_ty(t) {
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
                    //                 found closure `[closure@$DIR/issue-20862.rs:2:5: 2:14 x:_]`
                    if !self.ignore_span.overlaps(span) {
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
        let (expected_found, exp_found, is_simple_error) = match values {
            None => (None, Mismatch::Fixed("type"), false),
            Some(values) => {
                let (is_simple_error, exp_found) = match values {
                    ValuePairs::Types(exp_found) => {
                        let is_simple_err =
                            exp_found.expected.is_simple_text() && exp_found.found.is_simple_text();
                        OpaqueTypesVisitor::visit_expected_found(
                            self.tcx,
                            exp_found.expected,
                            exp_found.found,
                            span,
                        )
                        .report(diag);

                        (is_simple_err, Mismatch::Variable(exp_found))
                    }
                    ValuePairs::TraitRefs(_) => (false, Mismatch::Fixed("trait")),
                    _ => (false, Mismatch::Fixed("type")),
                };
                let vals = match self.values_str(values) {
                    Some((expected, found)) => Some((expected, found)),
                    None => {
                        // Derived error. Cancel the emitter.
                        diag.cancel();
                        return;
                    }
                };
                (vals, exp_found, is_simple_error)
            }
        };

        // Ignore msg for object safe coercion
        // since E0038 message will be printed
        match terr {
            TypeError::ObjectUnsafeCoercion(_) => {}
            _ => {
                diag.span_label(span, terr.to_string());
                if let Some((sp, msg)) = secondary_span {
                    diag.span_label(sp, msg);
                }
            }
        };
        if let Some((expected, found)) = expected_found {
            let expected_label = match exp_found {
                Mismatch::Variable(ef) => ef.expected.prefix_string(),
                Mismatch::Fixed(s) => s.into(),
            };
            let found_label = match exp_found {
                Mismatch::Variable(ef) => ef.found.prefix_string(),
                Mismatch::Fixed(s) => s.into(),
            };
            let exp_found = match exp_found {
                Mismatch::Variable(exp_found) => Some(exp_found),
                Mismatch::Fixed(_) => None,
            };
            match (&terr, expected == found) {
                (TypeError::Sorts(values), extra) => {
                    let sort_string = |ty: Ty<'tcx>| match (extra, ty.kind()) {
                        (true, ty::Opaque(def_id, _)) => format!(
                            " (opaque type at {})",
                            self.tcx
                                .sess
                                .source_map()
                                .mk_substr_filename(self.tcx.def_span(*def_id)),
                        ),
                        (true, _) => format!(" ({})", ty.sort_string(self.tcx)),
                        (false, _) => "".to_string(),
                    };
                    if !(values.expected.is_simple_text() && values.found.is_simple_text())
                        || (exp_found.map_or(false, |ef| {
                            // This happens when the type error is a subset of the expectation,
                            // like when you have two references but one is `usize` and the other
                            // is `f32`. In those cases we still want to show the `note`. If the
                            // value from `ef` is `Infer(_)`, then we ignore it.
                            if !ef.expected.is_ty_infer() {
                                ef.expected != values.expected
                            } else if !ef.found.is_ty_infer() {
                                ef.found != values.found
                            } else {
                                false
                            }
                        }))
                    {
                        diag.note_expected_found_extra(
                            &expected_label,
                            expected,
                            &found_label,
                            found,
                            &sort_string(values.expected),
                            &sort_string(values.found),
                        );
                    }
                }
                (TypeError::ObjectUnsafeCoercion(_), _) => {
                    diag.note_unsuccessful_coercion(found, expected);
                }
                (_, _) => {
                    debug!(
                        "note_type_err: exp_found={:?}, expected={:?} found={:?}",
                        exp_found, expected, found
                    );
                    if !is_simple_error || terr.must_include_note() {
                        diag.note_expected_found(&expected_label, expected, &found_label, found);
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
                if exp_found.map_or(false, |ef| terr.found == ef.found) =>
            {
                Some(*terr)
            }
            _ => exp_found,
        };
        debug!("exp_found {:?} terr {:?}", exp_found, terr);
        if let Some(exp_found) = exp_found {
            self.suggest_as_ref_where_appropriate(span, &exp_found, diag);
            self.suggest_await_on_expect_found(cause, span, &exp_found, diag);
        }

        // In some (most?) cases cause.body_id points to actual body, but in some cases
        // it's a actual definition. According to the comments (e.g. in
        // librustc_typeck/check/compare_method.rs:compare_predicate_entailment) the latter
        // is relied upon by some other code. This might (or might not) need cleanup.
        let body_owner_def_id =
            self.tcx.hir().opt_local_def_id(cause.body_id).unwrap_or_else(|| {
                self.tcx.hir().body_owner_def_id(hir::BodyId { hir_id: cause.body_id })
            });
        self.check_and_note_conflicting_crates(diag, terr);
        self.tcx.note_and_explain_type_err(diag, terr, cause, span, body_owner_def_id.to_def_id());

        if let Some(ValuePairs::PolyTraitRefs(exp_found)) = values {
            if let ty::Closure(def_id, _) = exp_found.expected.skip_binder().self_ty().kind() {
                if let Some(def_id) = def_id.as_local() {
                    let hir_id = self.tcx.hir().local_def_id_to_hir_id(def_id);
                    let span = self.tcx.hir().span(hir_id);
                    diag.span_note(span, "this closure does not fulfill the lifetime requirements");
                }
            }
        }

        // It reads better to have the error origin as the final
        // thing.
        self.note_error_origin(diag, cause, exp_found);
    }

    pub fn get_impl_future_output_ty(&self, ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
        if let ty::Opaque(def_id, substs) = ty.kind() {
            let future_trait = self.tcx.require_lang_item(LangItem::Future, None);
            // Future::Output
            let item_def_id = self
                .tcx
                .associated_items(future_trait)
                .in_definition_order()
                .next()
                .unwrap()
                .def_id;

            let bounds = self.tcx.explicit_item_bounds(*def_id);

            for (predicate, _) in bounds {
                let predicate = predicate.subst(self.tcx, substs);
                if let ty::PredicateKind::Projection(projection_predicate) =
                    predicate.kind().skip_binder()
                {
                    if projection_predicate.projection_ty.item_def_id == item_def_id {
                        // We don't account for multiple `Future::Output = Ty` contraints.
                        return Some(projection_predicate.ty);
                    }
                }
            }
        }
        None
    }

    /// A possible error is to forget to add `.await` when using futures:
    ///
    /// ```
    /// async fn make_u32() -> u32 {
    ///     22
    /// }
    ///
    /// fn take_u32(x: u32) {}
    ///
    /// async fn foo() {
    ///     let x = make_u32();
    ///     take_u32(x);
    /// }
    /// ```
    ///
    /// This routine checks if the found type `T` implements `Future<Output=U>` where `U` is the
    /// expected type. If this is the case, and we are inside of an async body, it suggests adding
    /// `.await` to the tail of the expression.
    fn suggest_await_on_expect_found(
        &self,
        cause: &ObligationCause<'tcx>,
        exp_span: Span,
        exp_found: &ty::error::ExpectedFound<Ty<'tcx>>,
        diag: &mut DiagnosticBuilder<'tcx>,
    ) {
        debug!(
            "suggest_await_on_expect_found: exp_span={:?}, expected_ty={:?}, found_ty={:?}",
            exp_span, exp_found.expected, exp_found.found,
        );

        if let ObligationCauseCode::CompareImplMethodObligation { .. } = &cause.code {
            return;
        }

        match (
            self.get_impl_future_output_ty(exp_found.expected),
            self.get_impl_future_output_ty(exp_found.found),
        ) {
            (Some(exp), Some(found)) if ty::TyS::same_type(exp, found) => match &cause.code {
                ObligationCauseCode::IfExpression(box IfExpressionCause { then, .. }) => {
                    diag.multipart_suggestion(
                        "consider `await`ing on both `Future`s",
                        vec![
                            (then.shrink_to_hi(), ".await".to_string()),
                            (exp_span.shrink_to_hi(), ".await".to_string()),
                        ],
                        Applicability::MaybeIncorrect,
                    );
                }
                ObligationCauseCode::MatchExpressionArm(box MatchExpressionArmCause {
                    prior_arms,
                    ..
                }) => {
                    if let [.., arm_span] = &prior_arms[..] {
                        diag.multipart_suggestion(
                            "consider `await`ing on both `Future`s",
                            vec![
                                (arm_span.shrink_to_hi(), ".await".to_string()),
                                (exp_span.shrink_to_hi(), ".await".to_string()),
                            ],
                            Applicability::MaybeIncorrect,
                        );
                    } else {
                        diag.help("consider `await`ing on both `Future`s");
                    }
                }
                _ => {
                    diag.help("consider `await`ing on both `Future`s");
                }
            },
            (_, Some(ty)) if ty::TyS::same_type(exp_found.expected, ty) => {
                let span = match cause.code {
                    // scrutinee's span
                    ObligationCauseCode::Pattern { span: Some(span), .. } => span,
                    _ => exp_span,
                };
                diag.span_suggestion_verbose(
                    span.shrink_to_hi(),
                    "consider `await`ing on the `Future`",
                    ".await".to_string(),
                    Applicability::MaybeIncorrect,
                );
            }
            (Some(ty), _) if ty::TyS::same_type(ty, exp_found.found) => {
                let span = match cause.code {
                    // scrutinee's span
                    ObligationCauseCode::Pattern { span: Some(span), .. } => span,
                    _ => exp_span,
                };
                diag.span_suggestion_verbose(
                    span.shrink_to_hi(),
                    "consider `await`ing on the `Future`",
                    ".await".to_string(),
                    Applicability::MaybeIncorrect,
                );
            }
            _ => {}
        }
    }

    /// When encountering a case where `.as_ref()` on a `Result` or `Option` would be appropriate,
    /// suggests it.
    fn suggest_as_ref_where_appropriate(
        &self,
        span: Span,
        exp_found: &ty::error::ExpectedFound<Ty<'tcx>>,
        diag: &mut DiagnosticBuilder<'tcx>,
    ) {
        if let (ty::Adt(exp_def, exp_substs), ty::Ref(_, found_ty, _)) =
            (exp_found.expected.kind(), exp_found.found.kind())
        {
            if let ty::Adt(found_def, found_substs) = *found_ty.kind() {
                let path_str = format!("{:?}", exp_def);
                if exp_def == &found_def {
                    let opt_msg = "you can convert from `&Option<T>` to `Option<&T>` using \
                                       `.as_ref()`";
                    let result_msg = "you can convert from `&Result<T, E>` to \
                                          `Result<&T, &E>` using `.as_ref()`";
                    let have_as_ref = &[
                        ("std::option::Option", opt_msg),
                        ("core::option::Option", opt_msg),
                        ("std::result::Result", result_msg),
                        ("core::result::Result", result_msg),
                    ];
                    if let Some(msg) = have_as_ref
                        .iter()
                        .find_map(|(path, msg)| (&path_str == path).then_some(msg))
                    {
                        let mut show_suggestion = true;
                        for (exp_ty, found_ty) in exp_substs.types().zip(found_substs.types()) {
                            match *exp_ty.kind() {
                                ty::Ref(_, exp_ty, _) => {
                                    match (exp_ty.kind(), found_ty.kind()) {
                                        (_, ty::Param(_))
                                        | (_, ty::Infer(_))
                                        | (ty::Param(_), _)
                                        | (ty::Infer(_), _) => {}
                                        _ if ty::TyS::same_type(exp_ty, found_ty) => {}
                                        _ => show_suggestion = false,
                                    };
                                }
                                ty::Param(_) | ty::Infer(_) => {}
                                _ => show_suggestion = false,
                            }
                        }
                        if let (Ok(snippet), true) =
                            (self.tcx.sess.source_map().span_to_snippet(span), show_suggestion)
                        {
                            diag.span_suggestion(
                                span,
                                msg,
                                format!("{}.as_ref()", snippet),
                                Applicability::MachineApplicable,
                            );
                        }
                    }
                }
            }
        }
    }

    pub fn report_and_explain_type_error(
        &self,
        trace: TypeTrace<'tcx>,
        terr: &TypeError<'tcx>,
    ) -> DiagnosticBuilder<'tcx> {
        debug!("report_and_explain_type_error(trace={:?}, terr={:?})", trace, terr);

        let span = trace.cause.span(self.tcx);
        let failure_code = trace.cause.as_failure_code(terr);
        let mut diag = match failure_code {
            FailureCode::Error0038(did) => {
                let violations = self.tcx.object_safety_violations(did);
                report_object_safety_error(self.tcx, span, did, violations)
            }
            FailureCode::Error0317(failure_str) => {
                struct_span_err!(self.tcx.sess, span, E0317, "{}", failure_str)
            }
            FailureCode::Error0580(failure_str) => {
                struct_span_err!(self.tcx.sess, span, E0580, "{}", failure_str)
            }
            FailureCode::Error0308(failure_str) => {
                struct_span_err!(self.tcx.sess, span, E0308, "{}", failure_str)
            }
            FailureCode::Error0644(failure_str) => {
                struct_span_err!(self.tcx.sess, span, E0644, "{}", failure_str)
            }
        };
        self.note_type_err(&mut diag, &trace.cause, None, Some(trace.values), terr);
        diag
    }

    fn values_str(
        &self,
        values: ValuePairs<'tcx>,
    ) -> Option<(DiagnosticStyledString, DiagnosticStyledString)> {
        match values {
            infer::Types(exp_found) => self.expected_found_str_ty(exp_found),
            infer::Regions(exp_found) => self.expected_found_str(exp_found),
            infer::Consts(exp_found) => self.expected_found_str(exp_found),
            infer::TraitRefs(exp_found) => {
                let pretty_exp_found = ty::error::ExpectedFound {
                    expected: exp_found.expected.print_only_trait_path(),
                    found: exp_found.found.print_only_trait_path(),
                };
                self.expected_found_str(pretty_exp_found)
            }
            infer::PolyTraitRefs(exp_found) => {
                let pretty_exp_found = ty::error::ExpectedFound {
                    expected: exp_found.expected.print_only_trait_path(),
                    found: exp_found.found.print_only_trait_path(),
                };
                self.expected_found_str(pretty_exp_found)
            }
        }
    }

    fn expected_found_str_ty(
        &self,
        exp_found: ty::error::ExpectedFound<Ty<'tcx>>,
    ) -> Option<(DiagnosticStyledString, DiagnosticStyledString)> {
        let exp_found = self.resolve_vars_if_possible(exp_found);
        if exp_found.references_error() {
            return None;
        }

        Some(self.cmp(exp_found.expected, exp_found.found))
    }

    /// Returns a string of the form "expected `{}`, found `{}`".
    fn expected_found_str<T: fmt::Display + TypeFoldable<'tcx>>(
        &self,
        exp_found: ty::error::ExpectedFound<T>,
    ) -> Option<(DiagnosticStyledString, DiagnosticStyledString)> {
        let exp_found = self.resolve_vars_if_possible(exp_found);
        if exp_found.references_error() {
            return None;
        }

        Some((
            DiagnosticStyledString::highlighted(exp_found.expected.to_string()),
            DiagnosticStyledString::highlighted(exp_found.found.to_string()),
        ))
    }

    pub fn report_generic_bound_failure(
        &self,
        span: Span,
        origin: Option<SubregionOrigin<'tcx>>,
        bound_kind: GenericKind<'tcx>,
        sub: Region<'tcx>,
    ) {
        self.construct_generic_bound_failure(span, origin, bound_kind, sub).emit();
    }

    pub fn construct_generic_bound_failure(
        &self,
        span: Span,
        origin: Option<SubregionOrigin<'tcx>>,
        bound_kind: GenericKind<'tcx>,
        sub: Region<'tcx>,
    ) -> DiagnosticBuilder<'a> {
        let hir = &self.tcx.hir();
        // Attempt to obtain the span of the parameter so we can
        // suggest adding an explicit lifetime bound to it.
        let generics = self
            .in_progress_typeck_results
            .map(|typeck_results| typeck_results.borrow().hir_owner)
            .map(|owner| {
                let hir_id = hir.local_def_id_to_hir_id(owner);
                let parent_id = hir.get_parent_item(hir_id);
                (
                    // Parent item could be a `mod`, so we check the HIR before calling:
                    if let Some(Node::Item(Item {
                        kind: ItemKind::Trait(..) | ItemKind::Impl { .. },
                        ..
                    })) = hir.find(parent_id)
                    {
                        Some(self.tcx.generics_of(hir.local_def_id(parent_id).to_def_id()))
                    } else {
                        None
                    },
                    self.tcx.generics_of(owner.to_def_id()),
                )
            });
        let type_param_span = match (generics, bound_kind) {
            (Some((_, ref generics)), GenericKind::Param(ref param)) => {
                // Account for the case where `param` corresponds to `Self`,
                // which doesn't have the expected type argument.
                if !(generics.has_self && param.index == 0) {
                    let type_param = generics.type_param(param, self.tcx);
                    type_param.def_id.as_local().map(|def_id| {
                        // Get the `hir::Param` to verify whether it already has any bounds.
                        // We do this to avoid suggesting code that ends up as `T: 'a'b`,
                        // instead we suggest `T: 'a + 'b` in that case.
                        let id = hir.local_def_id_to_hir_id(def_id);
                        let mut has_bounds = false;
                        if let Node::GenericParam(param) = hir.get(id) {
                            has_bounds = !param.bounds.is_empty();
                        }
                        let sp = hir.span(id);
                        // `sp` only covers `T`, change it so that it covers
                        // `T:` when appropriate
                        let is_impl_trait = bound_kind.to_string().starts_with("impl ");
                        let sp = if has_bounds && !is_impl_trait {
                            sp.to(self
                                .tcx
                                .sess
                                .source_map()
                                .next_point(self.tcx.sess.source_map().next_point(sp)))
                        } else {
                            sp
                        };
                        (sp, has_bounds, is_impl_trait)
                    })
                } else {
                    None
                }
            }
            _ => None,
        };
        let new_lt = generics
            .as_ref()
            .and_then(|(parent_g, g)| {
                let possible: Vec<_> = (b'a'..=b'z').map(|c| format!("'{}", c as char)).collect();
                let mut lts_names = g
                    .params
                    .iter()
                    .filter(|p| matches!(p.kind, ty::GenericParamDefKind::Lifetime))
                    .map(|p| p.name.as_str())
                    .collect::<Vec<_>>();
                if let Some(g) = parent_g {
                    lts_names.extend(
                        g.params
                            .iter()
                            .filter(|p| matches!(p.kind, ty::GenericParamDefKind::Lifetime))
                            .map(|p| p.name.as_str()),
                    );
                }
                let lts = lts_names.iter().map(|s| -> &str { &*s }).collect::<Vec<_>>();
                possible.into_iter().find(|candidate| !lts.contains(&candidate.as_str()))
            })
            .unwrap_or("'lt".to_string());
        let add_lt_sugg = generics
            .as_ref()
            .and_then(|(_, g)| g.params.first())
            .and_then(|param| param.def_id.as_local())
            .map(|def_id| {
                (
                    hir.span(hir.local_def_id_to_hir_id(def_id)).shrink_to_lo(),
                    format!("{}, ", new_lt),
                )
            });

        let labeled_user_string = match bound_kind {
            GenericKind::Param(ref p) => format!("the parameter type `{}`", p),
            GenericKind::Projection(ref p) => format!("the associated type `{}`", p),
        };

        if let Some(SubregionOrigin::CompareImplMethodObligation {
            span,
            item_name,
            impl_item_def_id,
            trait_item_def_id,
        }) = origin
        {
            return self.report_extra_impl_obligation(
                span,
                item_name,
                impl_item_def_id,
                trait_item_def_id,
                &format!("`{}: {}`", bound_kind, sub),
            );
        }

        fn binding_suggestion<'tcx, S: fmt::Display>(
            err: &mut DiagnosticBuilder<'tcx>,
            type_param_span: Option<(Span, bool, bool)>,
            bound_kind: GenericKind<'tcx>,
            sub: S,
        ) {
            let msg = "consider adding an explicit lifetime bound";
            if let Some((sp, has_lifetimes, is_impl_trait)) = type_param_span {
                let suggestion = if is_impl_trait {
                    format!("{} + {}", bound_kind, sub)
                } else {
                    let tail = if has_lifetimes { " + " } else { "" };
                    format!("{}: {}{}", bound_kind, sub, tail)
                };
                err.span_suggestion(
                    sp,
                    &format!("{}...", msg),
                    suggestion,
                    Applicability::MaybeIncorrect, // Issue #41966
                );
            } else {
                let consider = format!(
                    "{} {}...",
                    msg,
                    if type_param_span.map_or(false, |(_, _, is_impl_trait)| is_impl_trait) {
                        format!(" `{}` to `{}`", sub, bound_kind)
                    } else {
                        format!("`{}: {}`", bound_kind, sub)
                    },
                );
                err.help(&consider);
            }
        }

        let new_binding_suggestion =
            |err: &mut DiagnosticBuilder<'tcx>,
             type_param_span: Option<(Span, bool, bool)>,
             bound_kind: GenericKind<'tcx>| {
                let msg = "consider introducing an explicit lifetime bound";
                if let Some((sp, has_lifetimes, is_impl_trait)) = type_param_span {
                    let suggestion = if is_impl_trait {
                        (sp.shrink_to_hi(), format!(" + {}", new_lt))
                    } else {
                        let tail = if has_lifetimes { " +" } else { "" };
                        (sp, format!("{}: {}{}", bound_kind, new_lt, tail))
                    };
                    let mut sugg =
                        vec![suggestion, (span.shrink_to_hi(), format!(" + {}", new_lt))];
                    if let Some(lt) = add_lt_sugg {
                        sugg.push(lt);
                        sugg.rotate_right(1);
                    }
                    // `MaybeIncorrect` due to issue #41966.
                    err.multipart_suggestion(msg, sugg, Applicability::MaybeIncorrect);
                }
            };

        let mut err = match *sub {
            ty::ReEarlyBound(ty::EarlyBoundRegion { name, .. })
            | ty::ReFree(ty::FreeRegion { bound_region: ty::BrNamed(_, name), .. }) => {
                // Does the required lifetime have a nice name we can print?
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0309,
                    "{} may not live long enough",
                    labeled_user_string
                );
                // Explicitly use the name instead of `sub`'s `Display` impl. The `Display` impl
                // for the bound is not suitable for suggestions when `-Zverbose` is set because it
                // uses `Debug` output, so we handle it specially here so that suggestions are
                // always correct.
                binding_suggestion(&mut err, type_param_span, bound_kind, name);
                err
            }

            ty::ReStatic => {
                // Does the required lifetime have a nice name we can print?
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0310,
                    "{} may not live long enough",
                    labeled_user_string
                );
                binding_suggestion(&mut err, type_param_span, bound_kind, "'static");
                err
            }

            _ => {
                // If not, be less specific.
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0311,
                    "{} may not live long enough",
                    labeled_user_string
                );
                note_and_explain_region(
                    self.tcx,
                    &mut err,
                    &format!("{} must be valid for ", labeled_user_string),
                    sub,
                    "...",
                );
                if let Some(infer::RelateParamBound(_, t)) = origin {
                    let t = self.resolve_vars_if_possible(t);
                    match t.kind() {
                        // We've got:
                        // fn get_later<G, T>(g: G, dest: &mut T) -> impl FnOnce() + '_
                        // suggest:
                        // fn get_later<'a, G: 'a, T>(g: G, dest: &mut T) -> impl FnOnce() + '_ + 'a
                        ty::Closure(_, _substs) | ty::Opaque(_, _substs) => {
                            new_binding_suggestion(&mut err, type_param_span, bound_kind);
                        }
                        _ => {
                            binding_suggestion(&mut err, type_param_span, bound_kind, new_lt);
                        }
                    }
                }
                err
            }
        };

        if let Some(origin) = origin {
            self.note_region_origin(&mut err, &origin);
        }
        err
    }

    fn report_sub_sup_conflict(
        &self,
        var_origin: RegionVariableOrigin,
        sub_origin: SubregionOrigin<'tcx>,
        sub_region: Region<'tcx>,
        sup_origin: SubregionOrigin<'tcx>,
        sup_region: Region<'tcx>,
    ) {
        let mut err = self.report_inference_failure(var_origin);

        note_and_explain_region(
            self.tcx,
            &mut err,
            "first, the lifetime cannot outlive ",
            sup_region,
            "...",
        );

        debug!("report_sub_sup_conflict: var_origin={:?}", var_origin);
        debug!("report_sub_sup_conflict: sub_region={:?}", sub_region);
        debug!("report_sub_sup_conflict: sub_origin={:?}", sub_origin);
        debug!("report_sub_sup_conflict: sup_region={:?}", sup_region);
        debug!("report_sub_sup_conflict: sup_origin={:?}", sup_origin);

        if let (&infer::Subtype(ref sup_trace), &infer::Subtype(ref sub_trace)) =
            (&sup_origin, &sub_origin)
        {
            debug!("report_sub_sup_conflict: sup_trace={:?}", sup_trace);
            debug!("report_sub_sup_conflict: sub_trace={:?}", sub_trace);
            debug!("report_sub_sup_conflict: sup_trace.values={:?}", sup_trace.values);
            debug!("report_sub_sup_conflict: sub_trace.values={:?}", sub_trace.values);

            if let (Some((sup_expected, sup_found)), Some((sub_expected, sub_found))) =
                (self.values_str(sup_trace.values), self.values_str(sub_trace.values))
            {
                if sub_expected == sup_expected && sub_found == sup_found {
                    note_and_explain_region(
                        self.tcx,
                        &mut err,
                        "...but the lifetime must also be valid for ",
                        sub_region,
                        "...",
                    );
                    err.span_note(
                        sup_trace.cause.span,
                        &format!("...so that the {}", sup_trace.cause.as_requirement_str()),
                    );

                    err.note_expected_found(&"", sup_expected, &"", sup_found);
                    err.emit();
                    return;
                }
            }
        }

        self.note_region_origin(&mut err, &sup_origin);

        note_and_explain_region(
            self.tcx,
            &mut err,
            "but, the lifetime must be valid for ",
            sub_region,
            "...",
        );

        self.note_region_origin(&mut err, &sub_origin);
        err.emit();
    }

    /// Determine whether an error associated with the given span and definition
    /// should be treated as being caused by the implicit `From` conversion
    /// within `?` desugaring.
    pub fn is_try_conversion(&self, span: Span, trait_def_id: DefId) -> bool {
        span.is_desugaring(DesugaringKind::QuestionMark)
            && self.tcx.is_diagnostic_item(sym::from_trait, trait_def_id)
    }
}

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    fn report_inference_failure(
        &self,
        var_origin: RegionVariableOrigin,
    ) -> DiagnosticBuilder<'tcx> {
        let br_string = |br: ty::BoundRegionKind| {
            let mut s = match br {
                ty::BrNamed(_, name) => name.to_string(),
                _ => String::new(),
            };
            if !s.is_empty() {
                s.push(' ');
            }
            s
        };
        let var_description = match var_origin {
            infer::MiscVariable(_) => String::new(),
            infer::PatternRegion(_) => " for pattern".to_string(),
            infer::AddrOfRegion(_) => " for borrow expression".to_string(),
            infer::Autoref(_, _) => " for autoref".to_string(),
            infer::Coercion(_) => " for automatic coercion".to_string(),
            infer::LateBoundRegion(_, br, infer::FnCall) => {
                format!(" for lifetime parameter {}in function call", br_string(br))
            }
            infer::LateBoundRegion(_, br, infer::HigherRankedType) => {
                format!(" for lifetime parameter {}in generic type", br_string(br))
            }
            infer::LateBoundRegion(_, br, infer::AssocTypeProjection(def_id)) => format!(
                " for lifetime parameter {}in trait containing associated type `{}`",
                br_string(br),
                self.tcx.associated_item(def_id).ident
            ),
            infer::EarlyBoundRegion(_, name) => format!(" for lifetime parameter `{}`", name),
            infer::BoundRegionInCoherence(name) => {
                format!(" for lifetime parameter `{}` in coherence check", name)
            }
            infer::UpvarRegion(ref upvar_id, _) => {
                let var_name = self.tcx.hir().name(upvar_id.var_path.hir_id);
                format!(" for capture of `{}` by closure", var_name)
            }
            infer::NLL(..) => bug!("NLL variable found in lexical phase"),
        };

        struct_span_err!(
            self.tcx.sess,
            var_origin.span(),
            E0495,
            "cannot infer an appropriate lifetime{} due to conflicting requirements",
            var_description
        )
    }
}

enum FailureCode {
    Error0038(DefId),
    Error0317(&'static str),
    Error0580(&'static str),
    Error0308(&'static str),
    Error0644(&'static str),
}

trait ObligationCauseExt<'tcx> {
    fn as_failure_code(&self, terr: &TypeError<'tcx>) -> FailureCode;
    fn as_requirement_str(&self) -> &'static str;
}

impl<'tcx> ObligationCauseExt<'tcx> for ObligationCause<'tcx> {
    fn as_failure_code(&self, terr: &TypeError<'tcx>) -> FailureCode {
        use self::FailureCode::*;
        use crate::traits::ObligationCauseCode::*;
        match self.code {
            CompareImplMethodObligation { .. } => Error0308("method not compatible with trait"),
            CompareImplTypeObligation { .. } => Error0308("type not compatible with trait"),
            MatchExpressionArm(box MatchExpressionArmCause { source, .. }) => {
                Error0308(match source {
                    hir::MatchSource::IfLetDesugar { .. } => {
                        "`if let` arms have incompatible types"
                    }
                    hir::MatchSource::TryDesugar => {
                        "try expression alternatives have incompatible types"
                    }
                    _ => "`match` arms have incompatible types",
                })
            }
            IfExpression { .. } => Error0308("`if` and `else` have incompatible types"),
            IfExpressionWithNoElse => Error0317("`if` may be missing an `else` clause"),
            MainFunctionType => Error0580("`main` function has wrong type"),
            StartFunctionType => Error0308("`#[start]` function has wrong type"),
            IntrinsicType => Error0308("intrinsic has wrong type"),
            MethodReceiver => Error0308("mismatched `self` parameter type"),

            // In the case where we have no more specific thing to
            // say, also take a look at the error code, maybe we can
            // tailor to that.
            _ => match terr {
                TypeError::CyclicTy(ty) if ty.is_closure() || ty.is_generator() => {
                    Error0644("closure/generator type that references itself")
                }
                TypeError::IntrinsicCast => {
                    Error0308("cannot coerce intrinsics to function pointers")
                }
                TypeError::ObjectUnsafeCoercion(did) => Error0038(*did),
                _ => Error0308("mismatched types"),
            },
        }
    }

    fn as_requirement_str(&self) -> &'static str {
        use crate::traits::ObligationCauseCode::*;
        match self.code {
            CompareImplMethodObligation { .. } => "method type is compatible with trait",
            CompareImplTypeObligation { .. } => "associated type is compatible with trait",
            ExprAssignable => "expression is assignable",
            MatchExpressionArm(box MatchExpressionArmCause { source, .. }) => match source {
                hir::MatchSource::IfLetDesugar { .. } => "`if let` arms have compatible types",
                _ => "`match` arms have compatible types",
            },
            IfExpression { .. } => "`if` and `else` have incompatible types",
            IfExpressionWithNoElse => "`if` missing an `else` returns `()`",
            MainFunctionType => "`main` function has the correct type",
            StartFunctionType => "`#[start]` function has the correct type",
            IntrinsicType => "intrinsic has the correct type",
            MethodReceiver => "method receiver has the correct type",
            _ => "types are compatible",
        }
    }
}

/// This is a bare signal of what kind of type we're dealing with. `ty::TyKind` tracks
/// extra information about each type, but we only care about the category.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum TyCategory {
    Closure,
    Opaque,
    Generator,
    Foreign,
}

impl TyCategory {
    fn descr(&self) -> &'static str {
        match self {
            Self::Closure => "closure",
            Self::Opaque => "opaque type",
            Self::Generator => "generator",
            Self::Foreign => "foreign type",
        }
    }

    pub fn from_ty(ty: Ty<'_>) -> Option<(Self, DefId)> {
        match *ty.kind() {
            ty::Closure(def_id, _) => Some((Self::Closure, def_id)),
            ty::Opaque(def_id, _) => Some((Self::Opaque, def_id)),
            ty::Generator(def_id, ..) => Some((Self::Generator, def_id)),
            ty::Foreign(def_id) => Some((Self::Foreign, def_id)),
            _ => None,
        }
    }
}
