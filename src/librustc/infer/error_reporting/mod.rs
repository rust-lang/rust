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
use crate::infer::{self, SuppressRegionErrors};

use crate::hir;
use crate::hir::def_id::DefId;
use crate::hir::Node;
use crate::infer::opaque_types;
use crate::middle::region;
use crate::traits::{ObligationCause, ObligationCauseCode};
use crate::ty::error::TypeError;
use crate::ty::{self, subst::{Subst, SubstsRef}, Region, Ty, TyCtxt, TypeFoldable};
use errors::{Applicability, DiagnosticBuilder, DiagnosticStyledString};
use std::{cmp, fmt};
use syntax_pos::{Pos, Span};

mod note;

mod need_type_info;

pub mod nice_region_error;

impl<'tcx> TyCtxt<'tcx> {
    pub fn note_and_explain_region(
        self,
        region_scope_tree: &region::ScopeTree,
        err: &mut DiagnosticBuilder<'_>,
        prefix: &str,
        region: ty::Region<'tcx>,
        suffix: &str,
    ) {
        let (description, span) = match *region {
            ty::ReScope(scope) => {
                let new_string;
                let unknown_scope = || {
                    format!(
                        "{}unknown scope: {:?}{}.  Please report a bug.",
                        prefix, scope, suffix
                    )
                };
                let span = scope.span(self, region_scope_tree);
                let tag = match self.hir().find(scope.hir_id(region_scope_tree)) {
                    Some(Node::Block(_)) => "block",
                    Some(Node::Expr(expr)) => match expr.node {
                        hir::ExprKind::Call(..) => "call",
                        hir::ExprKind::MethodCall(..) => "method call",
                        hir::ExprKind::Match(.., hir::MatchSource::IfLetDesugar { .. }) => "if let",
                        hir::ExprKind::Match(.., hir::MatchSource::WhileLetDesugar) => "while let",
                        hir::ExprKind::Match(.., hir::MatchSource::ForLoopDesugar) => "for",
                        hir::ExprKind::Match(..) => "match",
                        _ => "expression",
                    },
                    Some(Node::Stmt(_)) => "statement",
                    Some(Node::Item(it)) => Self::item_scope_tag(&it),
                    Some(Node::TraitItem(it)) => Self::trait_item_scope_tag(&it),
                    Some(Node::ImplItem(it)) => Self::impl_item_scope_tag(&it),
                    Some(_) | None => {
                        err.span_note(span, &unknown_scope());
                        return;
                    }
                };
                let scope_decorated_tag = match scope.data {
                    region::ScopeData::Node => tag,
                    region::ScopeData::CallSite => "scope of call-site for function",
                    region::ScopeData::Arguments => "scope of function body",
                    region::ScopeData::Destruction => {
                        new_string = format!("destruction scope surrounding {}", tag);
                        &new_string[..]
                    }
                    region::ScopeData::Remainder(first_statement_index) => {
                        new_string = format!(
                            "block suffix following statement {}",
                            first_statement_index.index()
                        );
                        &new_string[..]
                    }
                };
                self.explain_span(scope_decorated_tag, span)
            }

            ty::ReEarlyBound(_) | ty::ReFree(_) | ty::ReStatic => {
                self.msg_span_from_free_region(region)
            }

            ty::ReEmpty => ("the empty lifetime".to_owned(), None),

            ty::RePlaceholder(_) => (format!("any other region"), None),

            // FIXME(#13998) RePlaceholder should probably print like
            // ReFree rather than dumping Debug output on the user.
            //
            // We shouldn't really be having unification failures with ReVar
            // and ReLateBound though.
            ty::ReVar(_) | ty::ReLateBound(..) | ty::ReErased => {
                (format!("lifetime {:?}", region), None)
            }

            // We shouldn't encounter an error message with ReClosureBound.
            ty::ReClosureBound(..) => {
                bug!("encountered unexpected ReClosureBound: {:?}", region,);
            }
        };

        TyCtxt::emit_msg_span(err, prefix, description, span, suffix);
    }

    pub fn note_and_explain_free_region(
        self,
        err: &mut DiagnosticBuilder<'_>,
        prefix: &str,
        region: ty::Region<'tcx>,
        suffix: &str,
    ) {
        let (description, span) = self.msg_span_from_free_region(region);

        TyCtxt::emit_msg_span(err, prefix, description, span, suffix);
    }

    fn msg_span_from_free_region(self, region: ty::Region<'tcx>) -> (String, Option<Span>) {
        match *region {
            ty::ReEarlyBound(_) | ty::ReFree(_) => {
                self.msg_span_from_early_bound_and_free_regions(region)
            }
            ty::ReStatic => ("the static lifetime".to_owned(), None),
            ty::ReEmpty => ("an empty lifetime".to_owned(), None),
            _ => bug!("{:?}", region),
        }
    }

    fn msg_span_from_early_bound_and_free_regions(
        self,
        region: ty::Region<'tcx>,
    ) -> (String, Option<Span>) {
        let cm = self.sess.source_map();

        let scope = region.free_region_binding_scope(self);
        let node = self.hir().as_local_hir_id(scope).unwrap_or(hir::DUMMY_HIR_ID);
        let tag = match self.hir().find(node) {
            Some(Node::Block(_)) | Some(Node::Expr(_)) => "body",
            Some(Node::Item(it)) => Self::item_scope_tag(&it),
            Some(Node::TraitItem(it)) => Self::trait_item_scope_tag(&it),
            Some(Node::ImplItem(it)) => Self::impl_item_scope_tag(&it),
            _ => unreachable!(),
        };
        let (prefix, span) = match *region {
            ty::ReEarlyBound(ref br) => {
                let mut sp = cm.def_span(self.hir().span(node));
                if let Some(param) = self.hir()
                    .get_generics(scope)
                    .and_then(|generics| generics.get_named(br.name))
                {
                    sp = param.span;
                }
                (format!("the lifetime {} as defined on", br.name), sp)
            }
            ty::ReFree(ty::FreeRegion {
                bound_region: ty::BoundRegion::BrNamed(_, name),
                ..
            }) => {
                let mut sp = cm.def_span(self.hir().span(node));
                if let Some(param) = self.hir()
                    .get_generics(scope)
                    .and_then(|generics| generics.get_named(name))
                {
                    sp = param.span;
                }
                (format!("the lifetime {} as defined on", name), sp)
            }
            ty::ReFree(ref fr) => match fr.bound_region {
                ty::BrAnon(idx) => (
                    format!("the anonymous lifetime #{} defined on", idx + 1),
                    self.hir().span(node),
                ),
                _ => (
                    format!("the lifetime {} as defined on", region),
                    cm.def_span(self.hir().span(node)),
                ),
            },
            _ => bug!(),
        };
        let (msg, opt_span) = self.explain_span(tag, span);
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

    fn item_scope_tag(item: &hir::Item) -> &'static str {
        match item.node {
            hir::ItemKind::Impl(..) => "impl",
            hir::ItemKind::Struct(..) => "struct",
            hir::ItemKind::Union(..) => "union",
            hir::ItemKind::Enum(..) => "enum",
            hir::ItemKind::Trait(..) => "trait",
            hir::ItemKind::Fn(..) => "function body",
            _ => "item",
        }
    }

    fn trait_item_scope_tag(item: &hir::TraitItem) -> &'static str {
        match item.node {
            hir::TraitItemKind::Method(..) => "method body",
            hir::TraitItemKind::Const(..) | hir::TraitItemKind::Type(..) => "associated item",
        }
    }

    fn impl_item_scope_tag(item: &hir::ImplItem) -> &'static str {
        match item.node {
            hir::ImplItemKind::Method(..) => "method body",
            hir::ImplItemKind::Const(..)
            | hir::ImplItemKind::Existential(..)
            | hir::ImplItemKind::Type(..) => "associated item",
        }
    }

    fn explain_span(self, heading: &str, span: Span) -> (String, Option<Span>) {
        let lo = self.sess.source_map().lookup_char_pos(span.lo());
        (
            format!("the {} at {}:{}", heading, lo.line, lo.col.to_usize() + 1),
            Some(span),
        )
    }
}

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    pub fn report_region_errors(
        &self,
        region_scope_tree: &region::ScopeTree,
        errors: &Vec<RegionResolutionError<'tcx>>,
        suppress: SuppressRegionErrors,
    ) {
        debug!(
            "report_region_errors(): {} errors to start, suppress = {:?}",
            errors.len(),
            suppress
        );

        if suppress.suppressed() {
            return;
        }

        // try to pre-process the errors, which will group some of them
        // together into a `ProcessedErrors` group:
        let errors = self.process_errors(errors);

        debug!(
            "report_region_errors: {} errors after preprocessing",
            errors.len()
        );

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
                            self.report_placeholder_failure(region_scope_tree, origin, sub, sup)
                                .emit();
                        } else {
                            self.report_concrete_failure(region_scope_tree, origin, sub, sup)
                                .emit();
                        }
                    }

                    RegionResolutionError::GenericBoundFailure(origin, param_ty, sub) => {
                        self.report_generic_bound_failure(
                            region_scope_tree,
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
                            self.report_placeholder_failure(
                                region_scope_tree,
                                sub_origin,
                                sub_r,
                                sup_r,
                            )
                                .emit();
                        } else if sup_r.is_placeholder() {
                            self.report_placeholder_failure(
                                region_scope_tree,
                                sup_origin,
                                sub_r,
                                sup_r,
                            )
                                .emit();
                        } else {
                            self.report_sub_sup_conflict(
                                region_scope_tree,
                                var_origin,
                                sub_origin,
                                sub_r,
                                sup_origin,
                                sup_r,
                            );
                        }
                    }

                    RegionResolutionError::MemberConstraintFailure {
                        opaque_type_def_id,
                        hidden_ty,
                        member_region,
                        span: _,
                        choice_regions: _,
                    } => {
                        let hidden_ty = self.resolve_vars_if_possible(&hidden_ty);
                        opaque_types::unexpected_hidden_region_diagnostic(
                            self.tcx,
                            Some(region_scope_tree),
                            opaque_type_def_id,
                            hidden_ty,
                            member_region,
                        ).emit();
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
        errors: &Vec<RegionResolutionError<'tcx>>,
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
                | RegionResolutionError::MemberConstraintFailure { .. } => false,
        };

        let mut errors = if errors.iter().all(|e| is_bound_failure(e)) {
            errors.clone()
        } else {
            errors
            .iter()
            .filter(|&e| !is_bound_failure(e))
            .cloned()
            .collect()
        };

        // sort the errors by span, for better error message stability.
        errors.sort_by_key(|u| match *u {
            RegionResolutionError::ConcreteFailure(ref sro, _, _) => sro.span(),
            RegionResolutionError::GenericBoundFailure(ref sro, _, _) => sro.span(),
            RegionResolutionError::SubSupConflict(_, ref rvo, _, _, _, _) => rvo.span(),
            RegionResolutionError::MemberConstraintFailure { span, .. } => span,
        });
        errors
    }

    /// Adds a note if the types come from similarly named crates
    fn check_and_note_conflicting_crates(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        terr: &TypeError<'tcx>,
        sp: Span,
    ) {
        use hir::def_id::CrateNum;
        use hir::map::DisambiguatedDefPathData;
        use ty::print::Printer;
        use ty::subst::Kind;

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

            fn print_region(
                self,
                _region: ty::Region<'_>,
            ) -> Result<Self::Region, Self::Error> {
                Err(NonTrivialPath)
            }

            fn print_type(
                self,
                _ty: Ty<'tcx>,
            ) -> Result<Self::Type, Self::Error> {
                Err(NonTrivialPath)
            }

            fn print_dyn_existential(
                self,
                _predicates: &'tcx ty::List<ty::ExistentialPredicate<'tcx>>,
            ) -> Result<Self::DynExistential, Self::Error> {
                Err(NonTrivialPath)
            }

            fn print_const(
                self,
                _ct: &'tcx ty::Const<'tcx>,
            ) -> Result<Self::Const, Self::Error> {
                Err(NonTrivialPath)
            }

            fn path_crate(
                self,
                cnum: CrateNum,
            ) -> Result<Self::Path, Self::Error> {
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
                path.push(disambiguated_data.data.as_interned_str().to_string());
                Ok(path)
            }
            fn path_generic_args(
                self,
                print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
                _args: &[Kind<'tcx>],
            ) -> Result<Self::Path, Self::Error> {
                print_prefix(self)
            }
        }

        let report_path_match = |err: &mut DiagnosticBuilder<'_>, did1: DefId, did2: DefId| {
            // Only external crates, if either is from a local
            // module we could have false positives
            if !(did1.is_local() || did2.is_local()) && did1.krate != did2.krate {
                let abs_path = |def_id| {
                    AbsolutePathPrinter { tcx: self.tcx }
                        .print_def_path(def_id, &[])
                };

                // We compare strings because DefPath can be different
                // for imported and non-imported crates
                let same_path = || -> Result<_, NonTrivialPath> {
                    Ok(
                        self.tcx.def_path_str(did1) == self.tcx.def_path_str(did2) ||
                        abs_path(did1)? == abs_path(did2)?
                    )
                };
                if same_path().unwrap_or(false) {
                    let crate_name = self.tcx.crate_name(did1.krate);
                    err.span_note(
                        sp,
                        &format!(
                            "Perhaps two different versions \
                             of crate `{}` are being used?",
                            crate_name
                        ),
                    );
                }
            }
        };
        match *terr {
            TypeError::Sorts(ref exp_found) => {
                // if they are both "path types", there's a chance of ambiguity
                // due to different versions of the same crate
                if let (&ty::Adt(exp_adt, _), &ty::Adt(found_adt, _))
                     = (&exp_found.expected.sty, &exp_found.found.sty)
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
            ObligationCauseCode::MatchExpressionArmPattern { span, ty } => {
                if ty.is_suggestable() {  // don't show type `_`
                    err.span_label(span, format!("this match expression has type `{}`", ty));
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
            ObligationCauseCode::MatchExpressionArm {
                source,
                ref prior_arms,
                last_ty,
                discrim_hir_id,
                ..
            } => match source {
                hir::MatchSource::IfLetDesugar { .. } => {
                    let msg = "`if let` arms have incompatible types";
                    err.span_label(cause.span, msg);
                }
                hir::MatchSource::TryDesugar => {
                    if let Some(ty::error::ExpectedFound { expected, .. }) = exp_found {
                        let discrim_expr = self.tcx.hir().expect_expr(discrim_hir_id);
                        let discrim_ty = if let hir::ExprKind::Call(_, args) = &discrim_expr.node {
                            let arg_expr = args.first().expect("try desugaring call w/out arg");
                            self.in_progress_tables.and_then(|tables| {
                                tables.borrow().expr_ty_opt(arg_expr)
                            })
                        } else {
                            bug!("try desugaring w/out call expr as discriminant");
                        };

                        match discrim_ty {
                            Some(ty) if expected == ty => {
                                let source_map = self.tcx.sess.source_map();
                                err.span_suggestion(
                                    source_map.end_point(cause.span),
                                    "try removing this `?`",
                                    "".to_string(),
                                    Applicability::MachineApplicable,
                                );
                            },
                            _ => {},
                        }
                    }
                }
                _ => {
                    let msg = "`match` arms have incompatible types";
                    err.span_label(cause.span, msg);
                    if prior_arms.len() <= 4 {
                        for sp in prior_arms {
                            err.span_label(*sp, format!(
                                "this is found to be of type `{}`",
                                self.resolve_vars_if_possible(&last_ty),
                            ));
                        }
                    } else if let Some(sp) = prior_arms.last() {
                        err.span_label(*sp, format!(
                            "this and all prior arms are found to be of type `{}`", last_ty,
                        ));
                    }
                }
            },
            ObligationCauseCode::IfExpression { then, outer, semicolon } => {
                err.span_label(then, "expected because of this");
                outer.map(|sp| err.span_label(sp, "if and else have incompatible types"));
                if let Some(sp) = semicolon {
                    err.span_suggestion_short(
                        sp,
                        "consider removing this semicolon",
                        String::new(),
                        Applicability::MachineApplicable,
                    );
                }
            }
            _ => (),
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
        let lifetimes = sub.regions()
            .map(|lifetime| {
                let s = lifetime.to_string();
                if s.is_empty() {
                    "'_".to_string()
                } else {
                    s
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        if !lifetimes.is_empty() {
            if sub.regions().count() < len {
                value.push_normal(lifetimes + &", ");
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
            //self.push_comma(&mut value, &mut other_value, len, i);
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
    /// ```norun
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
            if let &ty::Adt(def, _) = &ta.sty {
                let path_ = self.tcx.def_path_str(def.did.clone());
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
        let mut type_params = generics.params.iter().rev().filter_map(|param| match param.kind {
            ty::GenericParamDefKind::Lifetime => None,
            ty::GenericParamDefKind::Type { has_default, .. } => Some((param.def_id, has_default)),
            ty::GenericParamDefKind::Const => None, // FIXME(const_generics:defaults)
        }).peekable();
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

    /// Compares two given types, eliding parts that are the same between them and highlighting
    /// relevant differences, and return two representation of those types for highlighted printing.
    fn cmp(&self, t1: Ty<'tcx>, t2: Ty<'tcx>) -> (DiagnosticStyledString, DiagnosticStyledString) {
        fn equals<'tcx>(a: Ty<'tcx>, b: Ty<'tcx>) -> bool {
            match (&a.sty, &b.sty) {
                (a, b) if *a == *b => true,
                (&ty::Int(_), &ty::Infer(ty::InferTy::IntVar(_)))
                | (&ty::Infer(ty::InferTy::IntVar(_)), &ty::Int(_))
                | (&ty::Infer(ty::InferTy::IntVar(_)), &ty::Infer(ty::InferTy::IntVar(_)))
                | (&ty::Float(_), &ty::Infer(ty::InferTy::FloatVar(_)))
                | (&ty::Infer(ty::InferTy::FloatVar(_)), &ty::Float(_))
                | (&ty::Infer(ty::InferTy::FloatVar(_)), &ty::Infer(ty::InferTy::FloatVar(_))) => {
                    true
                }
                _ => false,
            }
        }

        fn push_ty_ref<'tcx>(
            r: &ty::Region<'tcx>,
            ty: Ty<'tcx>,
            mutbl: hir::Mutability,
            s: &mut DiagnosticStyledString,
        ) {
            let mut r = r.to_string();
            if r == "'_" {
                r.clear();
            } else {
                r.push(' ');
            }
            s.push_highlighted(format!(
                "&{}{}",
                r,
                if mutbl == hir::MutMutable { "mut " } else { "" }
            ));
            s.push_normal(ty.to_string());
        }

        match (&t1.sty, &t2.sty) {
            (&ty::Adt(def1, sub1), &ty::Adt(def2, sub2)) => {
                let sub_no_defaults_1 = self.strip_generic_default_params(def1.did, sub1);
                let sub_no_defaults_2 = self.strip_generic_default_params(def2.did, sub2);
                let mut values = (DiagnosticStyledString::new(), DiagnosticStyledString::new());
                let path1 = self.tcx.def_path_str(def1.did.clone());
                let path2 = self.tcx.def_path_str(def2.did.clone());
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

                    // Only draw `<...>` if there're lifetime/type arguments.
                    if len > 0 {
                        values.0.push_normal("<");
                        values.1.push_normal("<");
                    }

                    fn lifetime_display(lifetime: Region<'_>) -> String {
                        let s = lifetime.to_string();
                        if s.is_empty() {
                            "'_".to_string()
                        } else {
                            s
                        }
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
                        if l1 == l2 {
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
                    for (i, (ta1, ta2)) in type_arguments.take(len).enumerate() {
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
                    if self.cmp_type_arg(
                        &mut values.0,
                        &mut values.1,
                        path1.clone(),
                        sub_no_defaults_1,
                        path2.clone(),
                        &t2,
                    ).is_some()
                    {
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
                        &t1,
                    ).is_some()
                    {
                        return values;
                    }

                    // We couldn't find anything in common, highlight everything.
                    //     let x: Bar<Qux> = y::<Foo<Zar>>();
                    (
                        DiagnosticStyledString::highlighted(t1.to_string()),
                        DiagnosticStyledString::highlighted(t2.to_string()),
                    )
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

            _ => {
                if t1 == t2 {
                    // The two types are the same, elide and don't highlight.
                    (
                        DiagnosticStyledString::normal("_"),
                        DiagnosticStyledString::normal("_"),
                    )
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
        // For some types of errors, expected-found does not make
        // sense, so just ignore the values we were given.
        match terr {
            TypeError::CyclicTy(_) => {
                values = None;
            }
            _ => {}
        }

        let (expected_found, exp_found, is_simple_error) = match values {
            None => (None, None, false),
            Some(values) => {
                let (is_simple_error, exp_found) = match values {
                    ValuePairs::Types(exp_found) => {
                        let is_simple_err =
                            exp_found.expected.is_primitive() && exp_found.found.is_primitive();

                        (is_simple_err, Some(exp_found))
                    }
                    _ => (false, None),
                };
                let vals = match self.values_str(&values) {
                    Some((expected, found)) => Some((expected, found)),
                    None => {
                        // Derived error. Cancel the emitter.
                        self.tcx.sess.diagnostic().cancel(diag);
                        return;
                    }
                };
                (vals, exp_found, is_simple_error)
            }
        };

        let span = cause.span(self.tcx);

        diag.span_label(span, terr.to_string());
        if let Some((sp, msg)) = secondary_span {
            diag.span_label(sp, msg);
        }

        if let Some((expected, found)) = expected_found {
            match (terr, is_simple_error, expected == found) {
                (&TypeError::Sorts(ref values), false, true) => {
                    diag.note_expected_found_extra(
                        &"type",
                        expected,
                        found,
                        &format!(" ({})", values.expected.sort_string(self.tcx)),
                        &format!(" ({})", values.found.sort_string(self.tcx)),
                    );
                }
                (_, false, _) => {
                    if let Some(exp_found) = exp_found {
                        let (def_id, ret_ty) = match exp_found.found.sty {
                            ty::FnDef(def, _) => {
                                (Some(def), Some(self.tcx.fn_sig(def).output()))
                            }
                            _ => (None, None),
                        };

                        let exp_is_struct = match exp_found.expected.sty {
                            ty::Adt(def, _) => def.is_struct(),
                            _ => false,
                        };

                        if let (Some(def_id), Some(ret_ty)) = (def_id, ret_ty) {
                            if exp_is_struct && &exp_found.expected == ret_ty.skip_binder() {
                                let message = format!(
                                    "did you mean `{}(/* fields */)`?",
                                    self.tcx.def_path_str(def_id)
                                );
                                diag.span_label(span, message);
                            }
                        }
                        self.suggest_as_ref_where_appropriate(span, &exp_found, diag);
                    }

                    diag.note_expected_found(&"type", expected, found);
                }
                _ => (),
            }
        }

        self.check_and_note_conflicting_crates(diag, terr, span);
        self.tcx.note_and_explain_type_err(diag, terr, span);

        // It reads better to have the error origin as the final
        // thing.
        self.note_error_origin(diag, &cause, exp_found);
    }

    /// When encountering a case where `.as_ref()` on a `Result` or `Option` would be appropriate,
    /// suggest it.
    fn suggest_as_ref_where_appropriate(
        &self,
        span: Span,
        exp_found: &ty::error::ExpectedFound<Ty<'tcx>>,
        diag: &mut DiagnosticBuilder<'tcx>,
    ) {
        match (&exp_found.expected.sty, &exp_found.found.sty) {
            (ty::Adt(exp_def, exp_substs), ty::Ref(_, found_ty, _)) => {
                if let ty::Adt(found_def, found_substs) = found_ty.sty {
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
                        if let Some(msg) = have_as_ref.iter()
                            .filter_map(|(path, msg)| if &path_str == path {
                                Some(msg)
                            } else {
                                None
                            }).next()
                        {
                            let mut show_suggestion = true;
                            for (exp_ty, found_ty) in exp_substs.types().zip(found_substs.types()) {
                                match exp_ty.sty {
                                    ty::Ref(_, exp_ty, _) => {
                                        match (&exp_ty.sty, &found_ty.sty) {
                                            (_, ty::Param(_)) |
                                            (_, ty::Infer(_)) |
                                            (ty::Param(_), _) |
                                            (ty::Infer(_), _) => {}
                                            _ if ty::TyS::same_type(exp_ty, found_ty) => {}
                                            _ => show_suggestion = false,
                                        };
                                    }
                                    ty::Param(_) | ty::Infer(_) => {}
                                    _ => show_suggestion = false,
                                }
                            }
                            if let (Ok(snippet), true) = (
                                self.tcx.sess.source_map().span_to_snippet(span),
                                show_suggestion,
                            ) {
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
            _ => {}
        }
    }

    pub fn report_and_explain_type_error(
        &self,
        trace: TypeTrace<'tcx>,
        terr: &TypeError<'tcx>,
    ) -> DiagnosticBuilder<'tcx> {
        debug!(
            "report_and_explain_type_error(trace={:?}, terr={:?})",
            trace, terr
        );

        let span = trace.cause.span(self.tcx);
        let failure_code = trace.cause.as_failure_code(terr);
        let mut diag = match failure_code {
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
        values: &ValuePairs<'tcx>,
    ) -> Option<(DiagnosticStyledString, DiagnosticStyledString)> {
        match *values {
            infer::Types(ref exp_found) => self.expected_found_str_ty(exp_found),
            infer::Regions(ref exp_found) => self.expected_found_str(exp_found),
            infer::Consts(ref exp_found) => self.expected_found_str(exp_found),
            infer::TraitRefs(ref exp_found) => self.expected_found_str(exp_found),
            infer::PolyTraitRefs(ref exp_found) => self.expected_found_str(exp_found),
        }
    }

    fn expected_found_str_ty(
        &self,
        exp_found: &ty::error::ExpectedFound<Ty<'tcx>>,
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
        exp_found: &ty::error::ExpectedFound<T>,
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
        region_scope_tree: &region::ScopeTree,
        span: Span,
        origin: Option<SubregionOrigin<'tcx>>,
        bound_kind: GenericKind<'tcx>,
        sub: Region<'tcx>,
    ) {
        self.construct_generic_bound_failure(region_scope_tree, span, origin, bound_kind, sub)
            .emit()
    }

    pub fn construct_generic_bound_failure(
        &self,
        region_scope_tree: &region::ScopeTree,
        span: Span,
        origin: Option<SubregionOrigin<'tcx>>,
        bound_kind: GenericKind<'tcx>,
        sub: Region<'tcx>,
    ) -> DiagnosticBuilder<'a> {
        // Attempt to obtain the span of the parameter so we can
        // suggest adding an explicit lifetime bound to it.
        let type_param_span = match (self.in_progress_tables, bound_kind) {
            (Some(ref table), GenericKind::Param(ref param)) => {
                let table = table.borrow();
                table.local_id_root.and_then(|did| {
                    let generics = self.tcx.generics_of(did);
                    // Account for the case where `did` corresponds to `Self`, which doesn't have
                    // the expected type argument.
                    if !param.is_self() {
                        let type_param = generics.type_param(param, self.tcx);
                        let hir = &self.tcx.hir();
                        hir.as_local_hir_id(type_param.def_id).map(|id| {
                            // Get the `hir::Param` to verify whether it already has any bounds.
                            // We do this to avoid suggesting code that ends up as `T: 'a'b`,
                            // instead we suggest `T: 'a + 'b` in that case.
                            let mut has_bounds = false;
                            if let Node::GenericParam(ref param) = hir.get(id) {
                                has_bounds = !param.bounds.is_empty();
                            }
                            let sp = hir.span(id);
                            // `sp` only covers `T`, change it so that it covers
                            // `T:` when appropriate
                            let is_impl_trait = bound_kind.to_string().starts_with("impl ");
                            let sp = if has_bounds && !is_impl_trait {
                                sp.to(self.tcx
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
                })
            }
            _ => None,
        };

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
            let consider = format!(
                "consider adding an explicit lifetime bound {}",
                if type_param_span.map(|(_, _, is_impl_trait)| is_impl_trait).unwrap_or(false) {
                    format!(" `{}` to `{}`...", sub, bound_kind)
                } else {
                    format!("`{}: {}`...", bound_kind, sub)
                },
            );
            if let Some((sp, has_lifetimes, is_impl_trait)) = type_param_span {
                let suggestion = if is_impl_trait {
                    format!("{} + {}", bound_kind, sub)
                } else {
                    let tail = if has_lifetimes { " + " } else { "" };
                    format!("{}: {}{}", bound_kind, sub, tail)
                };
                err.span_suggestion_short(
                    sp,
                    &consider,
                    suggestion,
                    Applicability::MaybeIncorrect, // Issue #41966
                );
            } else {
                err.help(&consider);
            }
        }

        let mut err = match *sub {
            ty::ReEarlyBound(_)
            | ty::ReFree(ty::FreeRegion {
                bound_region: ty::BrNamed(..),
                ..
            }) => {
                // Does the required lifetime have a nice name we can print?
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0309,
                    "{} may not live long enough",
                    labeled_user_string
                );
                binding_suggestion(&mut err, type_param_span, bound_kind, sub);
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
                err.help(&format!(
                    "consider adding an explicit lifetime bound for `{}`",
                    bound_kind
                ));
                self.tcx.note_and_explain_region(
                    region_scope_tree,
                    &mut err,
                    &format!("{} must be valid for ", labeled_user_string),
                    sub,
                    "...",
                );
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
        region_scope_tree: &region::ScopeTree,
        var_origin: RegionVariableOrigin,
        sub_origin: SubregionOrigin<'tcx>,
        sub_region: Region<'tcx>,
        sup_origin: SubregionOrigin<'tcx>,
        sup_region: Region<'tcx>,
    ) {
        let mut err = self.report_inference_failure(var_origin);

        self.tcx.note_and_explain_region(
            region_scope_tree,
            &mut err,
            "first, the lifetime cannot outlive ",
            sup_region,
            "...",
        );

        match (&sup_origin, &sub_origin) {
            (&infer::Subtype(ref sup_trace), &infer::Subtype(ref sub_trace)) => {
                debug!("report_sub_sup_conflict: var_origin={:?}", var_origin);
                debug!("report_sub_sup_conflict: sub_region={:?}", sub_region);
                debug!("report_sub_sup_conflict: sub_origin={:?}", sub_origin);
                debug!("report_sub_sup_conflict: sup_region={:?}", sup_region);
                debug!("report_sub_sup_conflict: sup_origin={:?}", sup_origin);
                debug!("report_sub_sup_conflict: sup_trace={:?}", sup_trace);
                debug!("report_sub_sup_conflict: sub_trace={:?}", sub_trace);
                debug!("report_sub_sup_conflict: sup_trace.values={:?}", sup_trace.values);
                debug!("report_sub_sup_conflict: sub_trace.values={:?}", sub_trace.values);

                if let (Some((sup_expected, sup_found)), Some((sub_expected, sub_found))) = (
                    self.values_str(&sup_trace.values),
                    self.values_str(&sub_trace.values),
                ) {
                    if sub_expected == sup_expected && sub_found == sup_found {
                        self.tcx.note_and_explain_region(
                            region_scope_tree,
                            &mut err,
                            "...but the lifetime must also be valid for ",
                            sub_region,
                            "...",
                        );
                        err.note(&format!(
                            "...so that the {}:\nexpected {}\n   found {}",
                            sup_trace.cause.as_requirement_str(),
                            sup_expected.content(),
                            sup_found.content()
                        ));
                        err.emit();
                        return;
                    }
                }
            }
            _ => {}
        }

        self.note_region_origin(&mut err, &sup_origin);

        self.tcx.note_and_explain_region(
            region_scope_tree,
            &mut err,
            "but, the lifetime must be valid for ",
            sub_region,
            "...",
        );

        self.note_region_origin(&mut err, &sub_origin);
        err.emit();
    }
}

impl<'a, 'tcx> InferCtxt<'a, 'tcx> {
    fn report_inference_failure(
        &self,
        var_origin: RegionVariableOrigin,
    ) -> DiagnosticBuilder<'tcx> {
        let br_string = |br: ty::BoundRegion| {
            let mut s = match br {
                ty::BrNamed(_, name) => name.to_string(),
                _ => String::new(),
            };
            if !s.is_empty() {
                s.push_str(" ");
            }
            s
        };
        let var_description = match var_origin {
            infer::MiscVariable(_) => String::new(),
            infer::PatternRegion(_) => " for pattern".to_string(),
            infer::AddrOfRegion(_) => " for borrow expression".to_string(),
            infer::Autoref(_) => " for autoref".to_string(),
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
            "cannot infer an appropriate lifetime{} \
             due to conflicting requirements",
            var_description
        )
    }
}

enum FailureCode {
    Error0317(&'static str),
    Error0580(&'static str),
    Error0308(&'static str),
    Error0644(&'static str),
}

impl<'tcx> ObligationCause<'tcx> {
    fn as_failure_code(&self, terr: &TypeError<'tcx>) -> FailureCode {
        use self::FailureCode::*;
        use crate::traits::ObligationCauseCode::*;
        match self.code {
            CompareImplMethodObligation { .. } => Error0308("method not compatible with trait"),
            MatchExpressionArm { source, .. } => Error0308(match source {
                hir::MatchSource::IfLetDesugar { .. } => "`if let` arms have incompatible types",
                hir::MatchSource::TryDesugar => {
                    "try expression alternatives have incompatible types"
                }
                _ => "match arms have incompatible types",
            }),
            IfExpression { .. } => Error0308("if and else have incompatible types"),
            IfExpressionWithNoElse => Error0317("if may be missing an else clause"),
            MainFunctionType => Error0580("main function has wrong type"),
            StartFunctionType => Error0308("start function has wrong type"),
            IntrinsicType => Error0308("intrinsic has wrong type"),
            MethodReceiver => Error0308("mismatched method receiver"),

            // In the case where we have no more specific thing to
            // say, also take a look at the error code, maybe we can
            // tailor to that.
            _ => match terr {
                TypeError::CyclicTy(ty) if ty.is_closure() || ty.is_generator() => {
                    Error0644("closure/generator type that references itself")
                }
                _ => Error0308("mismatched types"),
            },
        }
    }

    fn as_requirement_str(&self) -> &'static str {
        use crate::traits::ObligationCauseCode::*;
        match self.code {
            CompareImplMethodObligation { .. } => "method type is compatible with trait",
            ExprAssignable => "expression is assignable",
            MatchExpressionArm { source, .. } => match source {
                hir::MatchSource::IfLetDesugar { .. } => "`if let` arms have compatible types",
                _ => "match arms have compatible types",
            },
            IfExpression { .. } => "if and else have compatible types",
            IfExpressionWithNoElse => "if missing an else returns ()",
            MainFunctionType => "`main` function has the correct type",
            StartFunctionType => "`start` function has the correct type",
            IntrinsicType => "intrinsic has the correct type",
            MethodReceiver => "method receiver has the correct type",
            _ => "types are compatible",
        }
    }
}
