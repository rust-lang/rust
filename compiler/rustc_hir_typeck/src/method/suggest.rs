//! Give useful errors and suggestions to users when an item can't be
//! found or is otherwise invalid.

// ignore-tidy-filelength

use core::ops::ControlFlow;
use std::borrow::Cow;
use std::path::PathBuf;

use hir::Expr;
use rustc_ast::ast::Mutability;
use rustc_attr_data_structures::{AttributeKind, find_attr};
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::sorted_map::SortedMap;
use rustc_data_structures::unord::UnordSet;
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagStyledString, MultiSpan, StashKey, pluralize, struct_span_code_err,
};
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::lang_items::LangItem;
use rustc_hir::{self as hir, ExprKind, HirId, Node, PathSegment, QPath};
use rustc_infer::infer::{self, RegionVariableOrigin};
use rustc_middle::bug;
use rustc_middle::ty::fast_reject::{DeepRejectCtxt, TreatParams, simplify_type};
use rustc_middle::ty::print::{
    PrintTraitRefExt as _, with_crate_prefix, with_forced_trimmed_paths,
    with_no_visible_paths_if_doc_hidden,
};
use rustc_middle::ty::{self, GenericArgKind, IsSuggestable, Ty, TyCtxt, TypeVisitableExt};
use rustc_span::def_id::DefIdSet;
use rustc_span::{
    DUMMY_SP, ErrorGuaranteed, ExpnKind, FileName, Ident, MacroKind, Span, Symbol, edit_distance,
    kw, sym,
};
use rustc_trait_selection::error_reporting::traits::DefIdOrName;
use rustc_trait_selection::error_reporting::traits::on_unimplemented::OnUnimplementedNote;
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt as _;
use rustc_trait_selection::traits::{
    FulfillmentError, Obligation, ObligationCause, ObligationCauseCode, supertraits,
};
use tracing::{debug, info, instrument};

use super::probe::{AutorefOrPtrAdjustment, IsSuggestion, Mode, ProbeScope};
use super::{CandidateSource, MethodError, NoMatchData};
use crate::errors::{self, CandidateTraitNote, NoAssociatedItem};
use crate::{Expectation, FnCtxt};

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    fn is_slice_ty(&self, ty: Ty<'tcx>, span: Span) -> bool {
        self.autoderef(span, ty)
            .silence_errors()
            .any(|(ty, _)| matches!(ty.kind(), ty::Slice(..) | ty::Array(..)))
    }

    fn impl_into_iterator_should_be_iterator(
        &self,
        ty: Ty<'tcx>,
        span: Span,
        unsatisfied_predicates: &Vec<(
            ty::Predicate<'tcx>,
            Option<ty::Predicate<'tcx>>,
            Option<ObligationCause<'tcx>>,
        )>,
    ) -> bool {
        fn predicate_bounds_generic_param<'tcx>(
            predicate: ty::Predicate<'_>,
            generics: &'tcx ty::Generics,
            generic_param: &ty::GenericParamDef,
            tcx: TyCtxt<'tcx>,
        ) -> bool {
            if let ty::PredicateKind::Clause(ty::ClauseKind::Trait(trait_pred)) =
                predicate.kind().as_ref().skip_binder()
            {
                let ty::TraitPredicate { trait_ref: ty::TraitRef { args, .. }, .. } = trait_pred;
                if args.is_empty() {
                    return false;
                }
                let Some(arg_ty) = args[0].as_type() else {
                    return false;
                };
                let ty::Param(param) = *arg_ty.kind() else {
                    return false;
                };
                // Is `generic_param` the same as the arg for this trait predicate?
                generic_param.index == generics.type_param(param, tcx).index
            } else {
                false
            }
        }

        let is_iterator_predicate = |predicate: ty::Predicate<'tcx>| -> bool {
            if let ty::PredicateKind::Clause(ty::ClauseKind::Trait(trait_pred)) =
                predicate.kind().as_ref().skip_binder()
            {
                self.tcx.is_diagnostic_item(sym::Iterator, trait_pred.trait_ref.def_id)
                    // ignore unsatisfied predicates generated from trying to auto-ref ty (#127511)
                    && trait_pred.trait_ref.self_ty() == ty
            } else {
                false
            }
        };

        // Does the `ty` implement `IntoIterator`?
        let Some(into_iterator_trait) = self.tcx.get_diagnostic_item(sym::IntoIterator) else {
            return false;
        };
        let trait_ref = ty::TraitRef::new(self.tcx, into_iterator_trait, [ty]);
        let obligation = Obligation::new(self.tcx, self.misc(span), self.param_env, trait_ref);
        if !self.predicate_must_hold_modulo_regions(&obligation) {
            return false;
        }

        match *ty.peel_refs().kind() {
            ty::Param(param) => {
                let generics = self.tcx.generics_of(self.body_id);
                let generic_param = generics.type_param(param, self.tcx);
                for unsatisfied in unsatisfied_predicates.iter() {
                    // The parameter implements `IntoIterator`
                    // but it has called a method that requires it to implement `Iterator`
                    if predicate_bounds_generic_param(
                        unsatisfied.0,
                        generics,
                        generic_param,
                        self.tcx,
                    ) && is_iterator_predicate(unsatisfied.0)
                    {
                        return true;
                    }
                }
            }
            ty::Slice(..) | ty::Adt(..) | ty::Alias(ty::Opaque, _) => {
                for unsatisfied in unsatisfied_predicates.iter() {
                    if is_iterator_predicate(unsatisfied.0) {
                        return true;
                    }
                }
            }
            _ => return false,
        }
        false
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn report_method_error(
        &self,
        call_id: HirId,
        rcvr_ty: Ty<'tcx>,
        error: MethodError<'tcx>,
        expected: Expectation<'tcx>,
        trait_missing_method: bool,
    ) -> ErrorGuaranteed {
        // NOTE: Reporting a method error should also suppress any unused trait errors,
        // since the method error is very possibly the reason why the trait wasn't used.
        for &import_id in
            self.tcx.in_scope_traits(call_id).into_iter().flatten().flat_map(|c| &c.import_ids)
        {
            self.typeck_results.borrow_mut().used_trait_imports.insert(import_id);
        }

        let (span, expr_span, source, item_name, args) = match self.tcx.hir_node(call_id) {
            hir::Node::Expr(&hir::Expr {
                kind: hir::ExprKind::MethodCall(segment, rcvr, args, _),
                span,
                ..
            }) => {
                (segment.ident.span, span, SelfSource::MethodCall(rcvr), segment.ident, Some(args))
            }
            hir::Node::Expr(&hir::Expr {
                kind: hir::ExprKind::Path(QPath::TypeRelative(rcvr, segment)),
                span,
                ..
            })
            | hir::Node::PatExpr(&hir::PatExpr {
                kind: hir::PatExprKind::Path(QPath::TypeRelative(rcvr, segment)),
                span,
                ..
            })
            | hir::Node::Pat(&hir::Pat {
                kind:
                    hir::PatKind::Struct(QPath::TypeRelative(rcvr, segment), ..)
                    | hir::PatKind::TupleStruct(QPath::TypeRelative(rcvr, segment), ..),
                span,
                ..
            }) => {
                let args = match self.tcx.parent_hir_node(call_id) {
                    hir::Node::Expr(&hir::Expr {
                        kind: hir::ExprKind::Call(callee, args), ..
                    }) if callee.hir_id == call_id => Some(args),
                    _ => None,
                };
                (segment.ident.span, span, SelfSource::QPath(rcvr), segment.ident, args)
            }
            node => unreachable!("{node:?}"),
        };

        // Try to get the span of the identifier within the expression's syntax context
        // (if that's different).
        let within_macro_span = span.within_macro(expr_span, self.tcx.sess.source_map());

        // Avoid suggestions when we don't know what's going on.
        if let Err(guar) = rcvr_ty.error_reported() {
            return guar;
        }

        match error {
            MethodError::NoMatch(mut no_match_data) => self.report_no_match_method_error(
                span,
                rcvr_ty,
                item_name,
                call_id,
                source,
                args,
                expr_span,
                &mut no_match_data,
                expected,
                trait_missing_method,
                within_macro_span,
            ),

            MethodError::Ambiguity(mut sources) => {
                let mut err = struct_span_code_err!(
                    self.dcx(),
                    item_name.span,
                    E0034,
                    "multiple applicable items in scope"
                );
                err.span_label(item_name.span, format!("multiple `{item_name}` found"));
                if let Some(within_macro_span) = within_macro_span {
                    err.span_label(within_macro_span, "due to this macro variable");
                }

                self.note_candidates_on_method_error(
                    rcvr_ty,
                    item_name,
                    source,
                    args,
                    span,
                    &mut err,
                    &mut sources,
                    Some(expr_span),
                );
                err.emit()
            }

            MethodError::PrivateMatch(kind, def_id, out_of_scope_traits) => {
                let kind = self.tcx.def_kind_descr(kind, def_id);
                let mut err = struct_span_code_err!(
                    self.dcx(),
                    item_name.span,
                    E0624,
                    "{} `{}` is private",
                    kind,
                    item_name
                );
                err.span_label(item_name.span, format!("private {kind}"));
                let sp =
                    self.tcx.hir_span_if_local(def_id).unwrap_or_else(|| self.tcx.def_span(def_id));
                err.span_label(sp, format!("private {kind} defined here"));
                if let Some(within_macro_span) = within_macro_span {
                    err.span_label(within_macro_span, "due to this macro variable");
                }
                self.suggest_valid_traits(&mut err, item_name, out_of_scope_traits, true);
                err.emit()
            }

            MethodError::IllegalSizedBound { candidates, needs_mut, bound_span, self_expr } => {
                let msg = if needs_mut {
                    with_forced_trimmed_paths!(format!(
                        "the `{item_name}` method cannot be invoked on `{rcvr_ty}`"
                    ))
                } else {
                    format!("the `{item_name}` method cannot be invoked on a trait object")
                };
                let mut err = self.dcx().struct_span_err(span, msg);
                if !needs_mut {
                    err.span_label(bound_span, "this has a `Sized` requirement");
                }
                if let Some(within_macro_span) = within_macro_span {
                    err.span_label(within_macro_span, "due to this macro variable");
                }
                if !candidates.is_empty() {
                    let help = format!(
                        "{an}other candidate{s} {were} found in the following trait{s}",
                        an = if candidates.len() == 1 { "an" } else { "" },
                        s = pluralize!(candidates.len()),
                        were = pluralize!("was", candidates.len()),
                    );
                    self.suggest_use_candidates(
                        candidates,
                        |accessible_sugg, inaccessible_sugg, span| {
                            let suggest_for_access =
                                |err: &mut Diag<'_>, mut msg: String, sugg: Vec<_>| {
                                    msg += &format!(
                                        ", perhaps add a `use` for {one_of_them}:",
                                        one_of_them =
                                            if sugg.len() == 1 { "it" } else { "one_of_them" },
                                    );
                                    err.span_suggestions(
                                        span,
                                        msg,
                                        sugg,
                                        Applicability::MaybeIncorrect,
                                    );
                                };
                            let suggest_for_privacy =
                                |err: &mut Diag<'_>, mut msg: String, suggs: Vec<String>| {
                                    if let [sugg] = suggs.as_slice() {
                                        err.help(format!("\
                                            trait `{}` provides `{item_name}` is implemented but not reachable",
                                            sugg.trim(),
                                        ));
                                    } else {
                                        msg += &format!(" but {} not reachable", pluralize!("is", suggs.len()));
                                        err.span_suggestions(
                                            span,
                                            msg,
                                            suggs,
                                            Applicability::MaybeIncorrect,
                                        );
                                    }
                                };
                            if accessible_sugg.is_empty() {
                                // `inaccessible_sugg` must not be empty
                                suggest_for_privacy(&mut err, help, inaccessible_sugg);
                            } else if inaccessible_sugg.is_empty() {
                                suggest_for_access(&mut err, help, accessible_sugg);
                            } else {
                                suggest_for_access(&mut err, help.clone(), accessible_sugg);
                                suggest_for_privacy(&mut err, help, inaccessible_sugg);
                            }
                        },
                    );
                }
                if let ty::Ref(region, t_type, mutability) = rcvr_ty.kind() {
                    if needs_mut {
                        let trait_type =
                            Ty::new_ref(self.tcx, *region, *t_type, mutability.invert());
                        let msg = format!("you need `{trait_type}` instead of `{rcvr_ty}`");
                        let mut kind = &self_expr.kind;
                        while let hir::ExprKind::AddrOf(_, _, expr)
                        | hir::ExprKind::Unary(hir::UnOp::Deref, expr) = kind
                        {
                            kind = &expr.kind;
                        }
                        if let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = kind
                            && let hir::def::Res::Local(hir_id) = path.res
                            && let hir::Node::Pat(b) = self.tcx.hir_node(hir_id)
                            && let hir::Node::Param(p) = self.tcx.parent_hir_node(b.hir_id)
                            && let Some(decl) = self.tcx.parent_hir_node(p.hir_id).fn_decl()
                            && let Some(ty) = decl.inputs.iter().find(|ty| ty.span == p.ty_span)
                            && let hir::TyKind::Ref(_, mut_ty) = &ty.kind
                            && let hir::Mutability::Not = mut_ty.mutbl
                        {
                            err.span_suggestion_verbose(
                                mut_ty.ty.span.shrink_to_lo(),
                                msg,
                                "mut ",
                                Applicability::MachineApplicable,
                            );
                        } else {
                            err.help(msg);
                        }
                    }
                }
                err.emit()
            }

            MethodError::ErrorReported(guar) => guar,

            MethodError::BadReturnType => bug!("no return type expectations but got BadReturnType"),
        }
    }

    fn suggest_missing_writer(&self, rcvr_ty: Ty<'tcx>, rcvr_expr: &hir::Expr<'tcx>) -> Diag<'_> {
        let mut file = None;
        let mut err = struct_span_code_err!(
            self.dcx(),
            rcvr_expr.span,
            E0599,
            "cannot write into `{}`",
            self.tcx.short_string(rcvr_ty, &mut file),
        );
        *err.long_ty_path() = file;
        err.span_note(
            rcvr_expr.span,
            "must implement `io::Write`, `fmt::Write`, or have a `write_fmt` method",
        );
        if let ExprKind::Lit(_) = rcvr_expr.kind {
            err.span_help(
                rcvr_expr.span.shrink_to_lo(),
                "a writer is needed before this format string",
            );
        };
        err
    }

    fn suggest_use_shadowed_binding_with_method(
        &self,
        self_source: SelfSource<'tcx>,
        method_name: Ident,
        ty_str_reported: &str,
        err: &mut Diag<'_>,
    ) {
        #[derive(Debug)]
        struct LetStmt {
            ty_hir_id_opt: Option<hir::HirId>,
            binding_id: hir::HirId,
            span: Span,
            init_hir_id: hir::HirId,
        }

        // Used for finding suggest binding.
        // ```rust
        // earlier binding for suggesting:
        // let y = vec![1, 2];
        // now binding:
        // if let Some(y) = x {
        //     y.push(y);
        // }
        // ```
        struct LetVisitor<'a, 'tcx> {
            // Error binding which don't have `method_name`.
            binding_name: Symbol,
            binding_id: hir::HirId,
            // Used for check if the suggest binding has `method_name`.
            fcx: &'a FnCtxt<'a, 'tcx>,
            call_expr: &'tcx Expr<'tcx>,
            method_name: Ident,
            // Suggest the binding which is shallowed.
            sugg_let: Option<LetStmt>,
        }

        impl<'a, 'tcx> LetVisitor<'a, 'tcx> {
            // Check scope of binding.
            fn is_sub_scope(&self, sub_id: hir::ItemLocalId, super_id: hir::ItemLocalId) -> bool {
                let scope_tree = self.fcx.tcx.region_scope_tree(self.fcx.body_id);
                if let Some(sub_var_scope) = scope_tree.var_scope(sub_id)
                    && let Some(super_var_scope) = scope_tree.var_scope(super_id)
                    && scope_tree.is_subscope_of(sub_var_scope, super_var_scope)
                {
                    return true;
                }
                false
            }

            // Check if an earlier shadowed binding make `the receiver` of a MethodCall has the method.
            // If it does, record the earlier binding for subsequent notes.
            fn check_and_add_sugg_binding(&mut self, binding: LetStmt) -> bool {
                if !self.is_sub_scope(self.binding_id.local_id, binding.binding_id.local_id) {
                    return false;
                }

                // Get the earlier shadowed binding'ty and use it to check the method.
                if let Some(ty_hir_id) = binding.ty_hir_id_opt
                    && let Some(tyck_ty) = self.fcx.node_ty_opt(ty_hir_id)
                {
                    if self
                        .fcx
                        .lookup_probe_for_diagnostic(
                            self.method_name,
                            tyck_ty,
                            self.call_expr,
                            ProbeScope::TraitsInScope,
                            None,
                        )
                        .is_ok()
                    {
                        self.sugg_let = Some(binding);
                        return true;
                    } else {
                        return false;
                    }
                }

                // If the shadowed binding has an itializer expression,
                // use the initializer expression'ty to try to find the method again.
                // For example like:  `let mut x = Vec::new();`,
                // `Vec::new()` is the itializer expression.
                if let Some(self_ty) = self.fcx.node_ty_opt(binding.init_hir_id)
                    && self
                        .fcx
                        .lookup_probe_for_diagnostic(
                            self.method_name,
                            self_ty,
                            self.call_expr,
                            ProbeScope::TraitsInScope,
                            None,
                        )
                        .is_ok()
                {
                    self.sugg_let = Some(binding);
                    return true;
                }
                return false;
            }
        }

        impl<'v> Visitor<'v> for LetVisitor<'_, '_> {
            type Result = ControlFlow<()>;
            fn visit_stmt(&mut self, ex: &'v hir::Stmt<'v>) -> Self::Result {
                if let hir::StmtKind::Let(&hir::LetStmt { pat, ty, init, .. }) = ex.kind
                    && let hir::PatKind::Binding(_, binding_id, binding_name, ..) = pat.kind
                    && let Some(init) = init
                    && binding_name.name == self.binding_name
                    && binding_id != self.binding_id
                {
                    if self.check_and_add_sugg_binding(LetStmt {
                        ty_hir_id_opt: ty.map(|ty| ty.hir_id),
                        binding_id,
                        span: pat.span,
                        init_hir_id: init.hir_id,
                    }) {
                        return ControlFlow::Break(());
                    }
                    ControlFlow::Continue(())
                } else {
                    hir::intravisit::walk_stmt(self, ex)
                }
            }

            // Used for find the error binding.
            // When the visitor reaches this point, all the shadowed bindings
            // have been found, so the visitor ends.
            fn visit_pat(&mut self, p: &'v hir::Pat<'v>) -> Self::Result {
                match p.kind {
                    hir::PatKind::Binding(_, binding_id, binding_name, _) => {
                        if binding_name.name == self.binding_name && binding_id == self.binding_id {
                            return ControlFlow::Break(());
                        }
                    }
                    _ => {
                        let _ = intravisit::walk_pat(self, p);
                    }
                }
                ControlFlow::Continue(())
            }
        }

        if let SelfSource::MethodCall(rcvr) = self_source
            && let hir::ExprKind::Path(QPath::Resolved(_, path)) = rcvr.kind
            && let hir::def::Res::Local(recv_id) = path.res
            && let Some(segment) = path.segments.first()
        {
            let body = self.tcx.hir_body_owned_by(self.body_id);

            if let Node::Expr(call_expr) = self.tcx.parent_hir_node(rcvr.hir_id) {
                let mut let_visitor = LetVisitor {
                    fcx: self,
                    call_expr,
                    binding_name: segment.ident.name,
                    binding_id: recv_id,
                    method_name,
                    sugg_let: None,
                };
                let _ = let_visitor.visit_body(&body);
                if let Some(sugg_let) = let_visitor.sugg_let
                    && let Some(self_ty) = self.node_ty_opt(sugg_let.init_hir_id)
                {
                    let _sm = self.infcx.tcx.sess.source_map();
                    let rcvr_name = segment.ident.name;
                    let mut span = MultiSpan::from_span(sugg_let.span);
                    span.push_span_label(sugg_let.span,
                            format!("`{rcvr_name}` of type `{self_ty}` that has method `{method_name}` defined earlier here"));
                    span.push_span_label(
                        self.tcx.hir_span(recv_id),
                        format!(
                            "earlier `{rcvr_name}` shadowed here with type `{ty_str_reported}`"
                        ),
                    );
                    err.span_note(
                        span,
                        format!(
                            "there's an earlier shadowed binding `{rcvr_name}` of type `{self_ty}` \
                                    that has method `{method_name}` available"
                        ),
                    );
                }
            }
        }
    }

    fn report_no_match_method_error(
        &self,
        mut span: Span,
        rcvr_ty: Ty<'tcx>,
        item_ident: Ident,
        expr_id: hir::HirId,
        source: SelfSource<'tcx>,
        args: Option<&'tcx [hir::Expr<'tcx>]>,
        sugg_span: Span,
        no_match_data: &mut NoMatchData<'tcx>,
        expected: Expectation<'tcx>,
        trait_missing_method: bool,
        within_macro_span: Option<Span>,
    ) -> ErrorGuaranteed {
        let mode = no_match_data.mode;
        let tcx = self.tcx;
        let rcvr_ty = self.resolve_vars_if_possible(rcvr_ty);
        let mut ty_file = None;
        let (ty_str, short_ty_str) =
            if trait_missing_method && let ty::Dynamic(predicates, _, _) = rcvr_ty.kind() {
                (predicates.to_string(), with_forced_trimmed_paths!(predicates.to_string()))
            } else {
                (
                    tcx.short_string(rcvr_ty, &mut ty_file),
                    with_forced_trimmed_paths!(rcvr_ty.to_string()),
                )
            };
        let is_method = mode == Mode::MethodCall;
        let unsatisfied_predicates = &no_match_data.unsatisfied_predicates;
        let similar_candidate = no_match_data.similar_candidate;
        let item_kind = if is_method {
            "method"
        } else if rcvr_ty.is_enum() {
            "variant or associated item"
        } else {
            match (item_ident.as_str().chars().next(), rcvr_ty.is_fresh_ty()) {
                (Some(name), false) if name.is_lowercase() => "function or associated item",
                (Some(_), false) => "associated item",
                (Some(_), true) | (None, false) => "variant or associated item",
                (None, true) => "variant",
            }
        };

        // We could pass the file for long types into these two, but it isn't strictly necessary
        // given how targeted they are.
        if let Err(guar) = self.report_failed_method_call_on_range_end(
            tcx,
            rcvr_ty,
            source,
            span,
            item_ident,
            &short_ty_str,
            &mut ty_file,
        ) {
            return guar;
        }
        if let Err(guar) = self.report_failed_method_call_on_numerical_infer_var(
            tcx,
            rcvr_ty,
            source,
            span,
            item_kind,
            item_ident,
            &short_ty_str,
            &mut ty_file,
        ) {
            return guar;
        }
        span = item_ident.span;

        // Don't show generic arguments when the method can't be found in any implementation (#81576).
        let mut ty_str_reported = ty_str.clone();
        if let ty::Adt(_, generics) = rcvr_ty.kind() {
            if generics.len() > 0 {
                let mut autoderef = self.autoderef(span, rcvr_ty).silence_errors();
                let candidate_found = autoderef.any(|(ty, _)| {
                    if let ty::Adt(adt_def, _) = ty.kind() {
                        self.tcx
                            .inherent_impls(adt_def.did())
                            .into_iter()
                            .any(|def_id| self.associated_value(*def_id, item_ident).is_some())
                    } else {
                        false
                    }
                });
                let has_deref = autoderef.step_count() > 0;
                if !candidate_found && !has_deref && unsatisfied_predicates.is_empty() {
                    if let Some((path_string, _)) = ty_str.split_once('<') {
                        ty_str_reported = path_string.to_string();
                    }
                }
            }
        }

        let is_write = sugg_span.ctxt().outer_expn_data().macro_def_id.is_some_and(|def_id| {
            tcx.is_diagnostic_item(sym::write_macro, def_id)
                || tcx.is_diagnostic_item(sym::writeln_macro, def_id)
        }) && item_ident.name == sym::write_fmt;
        let mut err = if is_write && let SelfSource::MethodCall(rcvr_expr) = source {
            self.suggest_missing_writer(rcvr_ty, rcvr_expr)
        } else {
            let mut err = self.dcx().create_err(NoAssociatedItem {
                span,
                item_kind,
                item_ident,
                ty_prefix: if trait_missing_method {
                    // FIXME(mu001999) E0599 maybe not suitable here because it is for types
                    Cow::from("trait")
                } else {
                    rcvr_ty.prefix_string(self.tcx)
                },
                ty_str: ty_str_reported.clone(),
                trait_missing_method,
            });

            if is_method {
                self.suggest_use_shadowed_binding_with_method(
                    source,
                    item_ident,
                    &ty_str_reported,
                    &mut err,
                );
            }

            // Check if we wrote `Self::Assoc(1)` as if it were a tuple ctor.
            if let SelfSource::QPath(ty) = source
                && let hir::TyKind::Path(hir::QPath::Resolved(_, path)) = ty.kind
                && let Res::SelfTyAlias { alias_to: impl_def_id, .. } = path.res
                && let DefKind::Impl { .. } = self.tcx.def_kind(impl_def_id)
                && let Some(candidate) = tcx.associated_items(impl_def_id).find_by_ident_and_kind(
                    self.tcx,
                    item_ident,
                    ty::AssocTag::Type,
                    impl_def_id,
                )
                && let Some(adt_def) = tcx.type_of(candidate.def_id).skip_binder().ty_adt_def()
                && adt_def.is_struct()
                && adt_def.non_enum_variant().ctor_kind() == Some(CtorKind::Fn)
            {
                let def_path = tcx.def_path_str(adt_def.did());
                err.span_suggestion(
                    sugg_span,
                    format!("to construct a value of type `{}`, use the explicit path", def_path),
                    def_path,
                    Applicability::MachineApplicable,
                );
            }

            err
        };
        if tcx.sess.source_map().is_multiline(sugg_span) {
            err.span_label(sugg_span.with_hi(span.lo()), "");
        }
        if let Some(within_macro_span) = within_macro_span {
            err.span_label(within_macro_span, "due to this macro variable");
        }

        if rcvr_ty.references_error() {
            err.downgrade_to_delayed_bug();
        }

        if matches!(source, SelfSource::QPath(_)) && args.is_some() {
            self.find_builder_fn(&mut err, rcvr_ty, expr_id);
        }

        if tcx.ty_is_opaque_future(rcvr_ty) && item_ident.name == sym::poll {
            err.help(format!(
                "method `poll` found on `Pin<&mut {ty_str}>`, \
                see documentation for `std::pin::Pin`"
            ));
            err.help("self type must be pinned to call `Future::poll`, \
                see https://rust-lang.github.io/async-book/04_pinning/01_chapter.html#pinning-in-practice"
            );
        }

        if let Mode::MethodCall = mode
            && let SelfSource::MethodCall(cal) = source
        {
            self.suggest_await_before_method(
                &mut err,
                item_ident,
                rcvr_ty,
                cal,
                span,
                expected.only_has_type(self),
            );
        }
        if let Some(span) =
            tcx.resolutions(()).confused_type_with_std_module.get(&span.with_parent(None))
        {
            err.span_suggestion(
                span.shrink_to_lo(),
                "you are looking for the module in `std`, not the primitive type",
                "std::",
                Applicability::MachineApplicable,
            );
        }

        // on pointers, check if the method would exist on a reference
        if let SelfSource::MethodCall(rcvr_expr) = source
            && let ty::RawPtr(ty, ptr_mutbl) = *rcvr_ty.kind()
            && let Ok(pick) = self.lookup_probe_for_diagnostic(
                item_ident,
                Ty::new_ref(tcx, ty::Region::new_error_misc(tcx), ty, ptr_mutbl),
                self.tcx.hir_expect_expr(self.tcx.parent_hir_id(rcvr_expr.hir_id)),
                ProbeScope::TraitsInScope,
                None,
            )
            && let ty::Ref(_, _, sugg_mutbl) = *pick.self_ty.kind()
            && (sugg_mutbl.is_not() || ptr_mutbl.is_mut())
        {
            let (method, method_anchor) = match sugg_mutbl {
                Mutability::Not => {
                    let method_anchor = match ptr_mutbl {
                        Mutability::Not => "as_ref",
                        Mutability::Mut => "as_ref-1",
                    };
                    ("as_ref", method_anchor)
                }
                Mutability::Mut => ("as_mut", "as_mut"),
            };
            err.span_note(
                tcx.def_span(pick.item.def_id),
                format!("the method `{item_ident}` exists on the type `{ty}`", ty = pick.self_ty),
            );
            let mut_str = ptr_mutbl.ptr_str();
            err.note(format!(
                "you might want to use the unsafe method `<*{mut_str} T>::{method}` to get \
                an optional reference to the value behind the pointer"
            ));
            err.note(format!(
                "read the documentation for `<*{mut_str} T>::{method}` and ensure you satisfy its \
                safety preconditions before calling it to avoid undefined behavior: \
                https://doc.rust-lang.org/std/primitive.pointer.html#method.{method_anchor}"
            ));
        }

        let mut ty_span = match rcvr_ty.kind() {
            ty::Param(param_type) => {
                Some(param_type.span_from_generics(self.tcx, self.body_id.to_def_id()))
            }
            ty::Adt(def, _) if def.did().is_local() => Some(tcx.def_span(def.did())),
            _ => None,
        };

        if let SelfSource::MethodCall(rcvr_expr) = source {
            self.suggest_fn_call(&mut err, rcvr_expr, rcvr_ty, |output_ty| {
                let call_expr = self.tcx.hir_expect_expr(self.tcx.parent_hir_id(rcvr_expr.hir_id));
                let probe = self.lookup_probe_for_diagnostic(
                    item_ident,
                    output_ty,
                    call_expr,
                    ProbeScope::AllTraits,
                    expected.only_has_type(self),
                );
                probe.is_ok()
            });
            self.note_internal_mutation_in_method(
                &mut err,
                rcvr_expr,
                expected.to_option(self),
                rcvr_ty,
            );
        }

        let mut custom_span_label = false;

        let static_candidates = &mut no_match_data.static_candidates;

        // `static_candidates` may have same candidates appended by
        // inherent and extension, which may result in incorrect
        // diagnostic.
        static_candidates.dedup();

        if !static_candidates.is_empty() {
            err.note(
                "found the following associated functions; to be used as methods, \
                 functions must have a `self` parameter",
            );
            err.span_label(span, "this is an associated function, not a method");
            custom_span_label = true;
        }
        if static_candidates.len() == 1 {
            self.suggest_associated_call_syntax(
                &mut err,
                static_candidates,
                rcvr_ty,
                source,
                item_ident,
                args,
                sugg_span,
            );
            self.note_candidates_on_method_error(
                rcvr_ty,
                item_ident,
                source,
                args,
                span,
                &mut err,
                static_candidates,
                None,
            );
        } else if static_candidates.len() > 1 {
            self.note_candidates_on_method_error(
                rcvr_ty,
                item_ident,
                source,
                args,
                span,
                &mut err,
                static_candidates,
                Some(sugg_span),
            );
        }

        let mut bound_spans: SortedMap<Span, Vec<String>> = Default::default();
        let mut restrict_type_params = false;
        let mut suggested_derive = false;
        let mut unsatisfied_bounds = false;
        if item_ident.name == sym::count && self.is_slice_ty(rcvr_ty, span) {
            let msg = "consider using `len` instead";
            if let SelfSource::MethodCall(_expr) = source {
                err.span_suggestion_short(span, msg, "len", Applicability::MachineApplicable);
            } else {
                err.span_label(span, msg);
            }
            if let Some(iterator_trait) = self.tcx.get_diagnostic_item(sym::Iterator) {
                let iterator_trait = self.tcx.def_path_str(iterator_trait);
                err.note(format!(
                    "`count` is defined on `{iterator_trait}`, which `{rcvr_ty}` does not implement"
                ));
            }
        } else if self.impl_into_iterator_should_be_iterator(rcvr_ty, span, unsatisfied_predicates)
        {
            err.span_label(span, format!("`{rcvr_ty}` is not an iterator"));
            if !span.in_external_macro(self.tcx.sess.source_map()) {
                err.multipart_suggestion_verbose(
                    "call `.into_iter()` first",
                    vec![(span.shrink_to_lo(), format!("into_iter()."))],
                    Applicability::MaybeIncorrect,
                );
            }
            return err.emit();
        } else if !unsatisfied_predicates.is_empty() && matches!(rcvr_ty.kind(), ty::Param(_)) {
            // We special case the situation where we are looking for `_` in
            // `<TypeParam as _>::method` because otherwise the machinery will look for blanket
            // implementations that have unsatisfied trait bounds to suggest, leading us to claim
            // things like "we're looking for a trait with method `cmp`, both `Iterator` and `Ord`
            // have one, in order to implement `Ord` you need to restrict `TypeParam: FnPtr` so
            // that `impl<T: FnPtr> Ord for T` can apply", which is not what we want. We have a type
            // parameter, we want to directly say "`Ord::cmp` and `Iterator::cmp` exist, restrict
            // `TypeParam: Ord` or `TypeParam: Iterator`"". That is done further down when calling
            // `self.suggest_traits_to_import`, so we ignore the `unsatisfied_predicates`
            // suggestions.
        } else if !unsatisfied_predicates.is_empty() {
            let mut type_params = FxIndexMap::default();

            // Pick out the list of unimplemented traits on the receiver.
            // This is used for custom error messages with the `#[rustc_on_unimplemented]` attribute.
            let mut unimplemented_traits = FxIndexMap::default();
            let mut unimplemented_traits_only = true;
            for (predicate, _parent_pred, cause) in unsatisfied_predicates {
                if let (ty::PredicateKind::Clause(ty::ClauseKind::Trait(p)), Some(cause)) =
                    (predicate.kind().skip_binder(), cause.as_ref())
                {
                    if p.trait_ref.self_ty() != rcvr_ty {
                        // This is necessary, not just to keep the errors clean, but also
                        // because our derived obligations can wind up with a trait ref that
                        // requires a different param_env to be correctly compared.
                        continue;
                    }
                    unimplemented_traits.entry(p.trait_ref.def_id).or_insert((
                        predicate.kind().rebind(p),
                        Obligation {
                            cause: cause.clone(),
                            param_env: self.param_env,
                            predicate: *predicate,
                            recursion_depth: 0,
                        },
                    ));
                }
            }

            // Make sure that, if any traits other than the found ones were involved,
            // we don't report an unimplemented trait.
            // We don't want to say that `iter::Cloned` is not an iterator, just
            // because of some non-Clone item being iterated over.
            for (predicate, _parent_pred, _cause) in unsatisfied_predicates {
                match predicate.kind().skip_binder() {
                    ty::PredicateKind::Clause(ty::ClauseKind::Trait(p))
                        if unimplemented_traits.contains_key(&p.trait_ref.def_id) => {}
                    _ => {
                        unimplemented_traits_only = false;
                        break;
                    }
                }
            }

            let mut collect_type_param_suggestions =
                |self_ty: Ty<'tcx>, parent_pred: ty::Predicate<'tcx>, obligation: &str| {
                    // We don't care about regions here, so it's fine to skip the binder here.
                    if let (ty::Param(_), ty::PredicateKind::Clause(ty::ClauseKind::Trait(p))) =
                        (self_ty.kind(), parent_pred.kind().skip_binder())
                    {
                        let node = match p.trait_ref.self_ty().kind() {
                            ty::Param(_) => {
                                // Account for `fn` items like in `issue-35677.rs` to
                                // suggest restricting its type params.
                                Some(self.tcx.hir_node_by_def_id(self.body_id))
                            }
                            ty::Adt(def, _) => def
                                .did()
                                .as_local()
                                .map(|def_id| self.tcx.hir_node_by_def_id(def_id)),
                            _ => None,
                        };
                        if let Some(hir::Node::Item(hir::Item { kind, .. })) = node
                            && let Some(g) = kind.generics()
                        {
                            let key = (
                                g.tail_span_for_predicate_suggestion(),
                                g.add_where_or_trailing_comma(),
                            );
                            type_params
                                .entry(key)
                                .or_insert_with(UnordSet::default)
                                .insert(obligation.to_owned());
                            return true;
                        }
                    }
                    false
                };
            let mut bound_span_label = |self_ty: Ty<'_>, obligation: &str, quiet: &str| {
                let msg = format!("`{}`", if obligation.len() > 50 { quiet } else { obligation });
                match self_ty.kind() {
                    // Point at the type that couldn't satisfy the bound.
                    ty::Adt(def, _) => {
                        bound_spans.get_mut_or_insert_default(tcx.def_span(def.did())).push(msg)
                    }
                    // Point at the trait object that couldn't satisfy the bound.
                    ty::Dynamic(preds, _, _) => {
                        for pred in preds.iter() {
                            match pred.skip_binder() {
                                ty::ExistentialPredicate::Trait(tr) => {
                                    bound_spans
                                        .get_mut_or_insert_default(tcx.def_span(tr.def_id))
                                        .push(msg.clone());
                                }
                                ty::ExistentialPredicate::Projection(_)
                                | ty::ExistentialPredicate::AutoTrait(_) => {}
                            }
                        }
                    }
                    // Point at the closure that couldn't satisfy the bound.
                    ty::Closure(def_id, _) => {
                        bound_spans
                            .get_mut_or_insert_default(tcx.def_span(*def_id))
                            .push(format!("`{quiet}`"));
                    }
                    _ => {}
                }
            };
            let mut format_pred = |pred: ty::Predicate<'tcx>| {
                let bound_predicate = pred.kind();
                match bound_predicate.skip_binder() {
                    ty::PredicateKind::Clause(ty::ClauseKind::Projection(pred)) => {
                        let pred = bound_predicate.rebind(pred);
                        // `<Foo as Iterator>::Item = String`.
                        let projection_term = pred.skip_binder().projection_term;
                        let quiet_projection_term =
                            projection_term.with_self_ty(tcx, Ty::new_var(tcx, ty::TyVid::ZERO));

                        let term = pred.skip_binder().term;

                        let obligation = format!("{projection_term} = {term}");
                        let quiet = with_forced_trimmed_paths!(format!(
                            "{} = {}",
                            quiet_projection_term, term
                        ));

                        bound_span_label(projection_term.self_ty(), &obligation, &quiet);
                        Some((obligation, projection_term.self_ty()))
                    }
                    ty::PredicateKind::Clause(ty::ClauseKind::Trait(poly_trait_ref)) => {
                        let p = poly_trait_ref.trait_ref;
                        let self_ty = p.self_ty();
                        let path = p.print_only_trait_path();
                        let obligation = format!("{self_ty}: {path}");
                        let quiet = with_forced_trimmed_paths!(format!("_: {}", path));
                        bound_span_label(self_ty, &obligation, &quiet);
                        Some((obligation, self_ty))
                    }
                    _ => None,
                }
            };

            // Find all the requirements that come from a local `impl` block.
            let mut skip_list: UnordSet<_> = Default::default();
            let mut spanned_predicates = FxIndexMap::default();
            for (p, parent_p, cause) in unsatisfied_predicates {
                // Extract the predicate span and parent def id of the cause,
                // if we have one.
                let (item_def_id, cause_span) = match cause.as_ref().map(|cause| cause.code()) {
                    Some(ObligationCauseCode::ImplDerived(data)) => {
                        (data.impl_or_alias_def_id, data.span)
                    }
                    Some(
                        ObligationCauseCode::WhereClauseInExpr(def_id, span, _, _)
                        | ObligationCauseCode::WhereClause(def_id, span),
                    ) if !span.is_dummy() => (*def_id, *span),
                    _ => continue,
                };

                // Don't point out the span of `WellFormed` predicates.
                if !matches!(
                    p.kind().skip_binder(),
                    ty::PredicateKind::Clause(
                        ty::ClauseKind::Projection(..) | ty::ClauseKind::Trait(..)
                    )
                ) {
                    continue;
                }

                match self.tcx.hir_get_if_local(item_def_id) {
                    // Unmet obligation comes from a `derive` macro, point at it once to
                    // avoid multiple span labels pointing at the same place.
                    Some(Node::Item(hir::Item {
                        kind: hir::ItemKind::Impl(hir::Impl { of_trait, self_ty, .. }),
                        ..
                    })) if matches!(
                        self_ty.span.ctxt().outer_expn_data().kind,
                        ExpnKind::Macro(MacroKind::Derive, _)
                    ) || matches!(
                        of_trait.as_ref().map(|t| t.path.span.ctxt().outer_expn_data().kind),
                        Some(ExpnKind::Macro(MacroKind::Derive, _))
                    ) =>
                    {
                        let span = self_ty.span.ctxt().outer_expn_data().call_site;
                        let entry = spanned_predicates.entry(span);
                        let entry = entry.or_insert_with(|| {
                            (FxIndexSet::default(), FxIndexSet::default(), Vec::new())
                        });
                        entry.0.insert(span);
                        entry.1.insert((
                            span,
                            "unsatisfied trait bound introduced in this `derive` macro",
                        ));
                        entry.2.push(p);
                        skip_list.insert(p);
                    }

                    // Unmet obligation coming from an `impl`.
                    Some(Node::Item(hir::Item {
                        kind: hir::ItemKind::Impl(hir::Impl { of_trait, self_ty, generics, .. }),
                        span: item_span,
                        ..
                    })) => {
                        let sized_pred =
                            unsatisfied_predicates.iter().any(|(pred, _, _)| {
                                match pred.kind().skip_binder() {
                                    ty::PredicateKind::Clause(ty::ClauseKind::Trait(pred)) => {
                                        self.tcx.is_lang_item(pred.def_id(), LangItem::Sized)
                                            && pred.polarity == ty::PredicatePolarity::Positive
                                    }
                                    _ => false,
                                }
                            });
                        for param in generics.params {
                            if param.span == cause_span && sized_pred {
                                let (sp, sugg) = match param.colon_span {
                                    Some(sp) => (sp.shrink_to_hi(), " ?Sized +"),
                                    None => (param.span.shrink_to_hi(), ": ?Sized"),
                                };
                                err.span_suggestion_verbose(
                                    sp,
                                    "consider relaxing the type parameter's implicit `Sized` bound",
                                    sugg,
                                    Applicability::MachineApplicable,
                                );
                            }
                        }
                        if let Some(pred) = parent_p {
                            // Done to add the "doesn't satisfy" `span_label`.
                            let _ = format_pred(*pred);
                        }
                        skip_list.insert(p);
                        let entry = spanned_predicates.entry(self_ty.span);
                        let entry = entry.or_insert_with(|| {
                            (FxIndexSet::default(), FxIndexSet::default(), Vec::new())
                        });
                        entry.2.push(p);
                        if cause_span != *item_span {
                            entry.0.insert(cause_span);
                            entry.1.insert((cause_span, "unsatisfied trait bound introduced here"));
                        } else {
                            if let Some(trait_ref) = of_trait {
                                entry.0.insert(trait_ref.path.span);
                            }
                            entry.0.insert(self_ty.span);
                        };
                        if let Some(trait_ref) = of_trait {
                            entry.1.insert((trait_ref.path.span, ""));
                        }
                        entry.1.insert((self_ty.span, ""));
                    }
                    Some(Node::Item(hir::Item {
                        kind: hir::ItemKind::Trait(rustc_ast::ast::IsAuto::Yes, ..),
                        span: item_span,
                        ..
                    })) => {
                        self.dcx().span_delayed_bug(
                            *item_span,
                            "auto trait is invoked with no method error, but no error reported?",
                        );
                    }
                    Some(
                        Node::Item(hir::Item {
                            kind:
                                hir::ItemKind::Trait(_, _, ident, ..)
                                | hir::ItemKind::TraitAlias(ident, ..),
                            ..
                        })
                        // We may also encounter unsatisfied GAT or method bounds
                        | Node::TraitItem(hir::TraitItem { ident, .. })
                        | Node::ImplItem(hir::ImplItem { ident, .. })
                    ) => {
                        skip_list.insert(p);
                        let entry = spanned_predicates.entry(ident.span);
                        let entry = entry.or_insert_with(|| {
                            (FxIndexSet::default(), FxIndexSet::default(), Vec::new())
                        });
                        entry.0.insert(cause_span);
                        entry.1.insert((ident.span, ""));
                        entry.1.insert((cause_span, "unsatisfied trait bound introduced here"));
                        entry.2.push(p);
                    }
                    _ => {
                        // It's possible to use well-formedness clauses to get obligations
                        // which point arbitrary items like ADTs, so there's no use in ICEing
                        // here if we find that the obligation originates from some other
                        // node that we don't handle.
                    }
                }
            }
            let mut spanned_predicates: Vec<_> = spanned_predicates.into_iter().collect();
            spanned_predicates.sort_by_key(|(span, _)| *span);
            for (_, (primary_spans, span_labels, predicates)) in spanned_predicates {
                let mut preds: Vec<_> = predicates
                    .iter()
                    .filter_map(|pred| format_pred(**pred))
                    .map(|(p, _)| format!("`{p}`"))
                    .collect();
                preds.sort();
                preds.dedup();
                let msg = if let [pred] = &preds[..] {
                    format!("trait bound {pred} was not satisfied")
                } else {
                    format!("the following trait bounds were not satisfied:\n{}", preds.join("\n"),)
                };
                let mut span: MultiSpan = primary_spans.into_iter().collect::<Vec<_>>().into();
                for (sp, label) in span_labels {
                    span.push_span_label(sp, label);
                }
                err.span_note(span, msg);
                unsatisfied_bounds = true;
            }

            let mut suggested_bounds = UnordSet::default();
            // The requirements that didn't have an `impl` span to show.
            let mut bound_list = unsatisfied_predicates
                .iter()
                .filter_map(|(pred, parent_pred, _cause)| {
                    let mut suggested = false;
                    format_pred(*pred).map(|(p, self_ty)| {
                        if let Some(parent) = parent_pred
                            && suggested_bounds.contains(parent)
                        {
                            // We don't suggest `PartialEq` when we already suggest `Eq`.
                        } else if !suggested_bounds.contains(pred)
                            && collect_type_param_suggestions(self_ty, *pred, &p)
                        {
                            suggested = true;
                            suggested_bounds.insert(pred);
                        }
                        (
                            match parent_pred {
                                None => format!("`{p}`"),
                                Some(parent_pred) => match format_pred(*parent_pred) {
                                    None => format!("`{p}`"),
                                    Some((parent_p, _)) => {
                                        if !suggested
                                            && !suggested_bounds.contains(pred)
                                            && !suggested_bounds.contains(parent_pred)
                                            && collect_type_param_suggestions(
                                                self_ty,
                                                *parent_pred,
                                                &p,
                                            )
                                        {
                                            suggested_bounds.insert(pred);
                                        }
                                        format!("`{p}`\nwhich is required by `{parent_p}`")
                                    }
                                },
                            },
                            *pred,
                        )
                    })
                })
                .filter(|(_, pred)| !skip_list.contains(&pred))
                .map(|(t, _)| t)
                .enumerate()
                .collect::<Vec<(usize, String)>>();

            if !matches!(rcvr_ty.peel_refs().kind(), ty::Param(_)) {
                for ((span, add_where_or_comma), obligations) in type_params.into_iter() {
                    restrict_type_params = true;
                    // #74886: Sort here so that the output is always the same.
                    let obligations = obligations.into_sorted_stable_ord();
                    err.span_suggestion_verbose(
                        span,
                        format!(
                            "consider restricting the type parameter{s} to satisfy the trait \
                             bound{s}",
                            s = pluralize!(obligations.len())
                        ),
                        format!("{} {}", add_where_or_comma, obligations.join(", ")),
                        Applicability::MaybeIncorrect,
                    );
                }
            }

            bound_list.sort_by(|(_, a), (_, b)| a.cmp(b)); // Sort alphabetically.
            bound_list.dedup_by(|(_, a), (_, b)| a == b); // #35677
            bound_list.sort_by_key(|(pos, _)| *pos); // Keep the original predicate order.

            if !bound_list.is_empty() || !skip_list.is_empty() {
                let bound_list =
                    bound_list.into_iter().map(|(_, path)| path).collect::<Vec<_>>().join("\n");
                let actual_prefix = rcvr_ty.prefix_string(self.tcx);
                info!("unimplemented_traits.len() == {}", unimplemented_traits.len());
                let (primary_message, label, notes) = if unimplemented_traits.len() == 1
                    && unimplemented_traits_only
                {
                    unimplemented_traits
                        .into_iter()
                        .next()
                        .map(|(_, (trait_ref, obligation))| {
                            if trait_ref.self_ty().references_error() || rcvr_ty.references_error()
                            {
                                // Avoid crashing.
                                return (None, None, Vec::new());
                            }
                            let OnUnimplementedNote { message, label, notes, .. } = self
                                .err_ctxt()
                                .on_unimplemented_note(trait_ref, &obligation, &mut ty_file);
                            (message, label, notes)
                        })
                        .unwrap()
                } else {
                    (None, None, Vec::new())
                };
                let primary_message = primary_message.unwrap_or_else(|| {
                    format!(
                        "the {item_kind} `{item_ident}` exists for {actual_prefix} `{ty_str}`, \
                         but its trait bounds were not satisfied"
                    )
                });
                err.primary_message(primary_message);
                if let Some(label) = label {
                    custom_span_label = true;
                    err.span_label(span, label);
                }
                if !bound_list.is_empty() {
                    err.note(format!(
                        "the following trait bounds were not satisfied:\n{bound_list}"
                    ));
                }
                for note in notes {
                    err.note(note);
                }

                suggested_derive = self.suggest_derive(&mut err, unsatisfied_predicates);

                unsatisfied_bounds = true;
            }
        } else if let ty::Adt(def, targs) = rcvr_ty.kind()
            && let SelfSource::MethodCall(rcvr_expr) = source
        {
            // This is useful for methods on arbitrary self types that might have a simple
            // mutability difference, like calling a method on `Pin<&mut Self>` that is on
            // `Pin<&Self>`.
            if targs.len() == 1 {
                let mut item_segment = hir::PathSegment::invalid();
                item_segment.ident = item_ident;
                for t in [Ty::new_mut_ref, Ty::new_imm_ref, |_, _, t| t] {
                    let new_args =
                        tcx.mk_args_from_iter(targs.iter().map(|arg| match arg.as_type() {
                            Some(ty) => ty::GenericArg::from(t(
                                tcx,
                                tcx.lifetimes.re_erased,
                                ty.peel_refs(),
                            )),
                            _ => arg,
                        }));
                    let rcvr_ty = Ty::new_adt(tcx, *def, new_args);
                    if let Ok(method) = self.lookup_method_for_diagnostic(
                        rcvr_ty,
                        &item_segment,
                        span,
                        tcx.parent_hir_node(rcvr_expr.hir_id).expect_expr(),
                        rcvr_expr,
                    ) {
                        err.span_note(
                            tcx.def_span(method.def_id),
                            format!("{item_kind} is available for `{rcvr_ty}`"),
                        );
                    }
                }
            }
        }

        let mut find_candidate_for_method = false;

        let mut label_span_not_found = |err: &mut Diag<'_>| {
            if unsatisfied_predicates.is_empty() {
                err.span_label(span, format!("{item_kind} not found in `{ty_str}`"));
                let is_string_or_ref_str = match rcvr_ty.kind() {
                    ty::Ref(_, ty, _) => {
                        ty.is_str()
                            || matches!(
                                ty.kind(),
                                ty::Adt(adt, _) if self.tcx.is_lang_item(adt.did(), LangItem::String)
                            )
                    }
                    ty::Adt(adt, _) => self.tcx.is_lang_item(adt.did(), LangItem::String),
                    _ => false,
                };
                if is_string_or_ref_str && item_ident.name == sym::iter {
                    err.span_suggestion_verbose(
                        item_ident.span,
                        "because of the in-memory representation of `&str`, to obtain \
                         an `Iterator` over each of its codepoint use method `chars`",
                        "chars",
                        Applicability::MachineApplicable,
                    );
                }
                if let ty::Adt(adt, _) = rcvr_ty.kind() {
                    let mut inherent_impls_candidate = self
                        .tcx
                        .inherent_impls(adt.did())
                        .into_iter()
                        .copied()
                        .filter(|def_id| {
                            if let Some(assoc) = self.associated_value(*def_id, item_ident) {
                                // Check for both mode is the same so we avoid suggesting
                                // incorrect associated item.
                                match (mode, assoc.is_method(), source) {
                                    (Mode::MethodCall, true, SelfSource::MethodCall(_)) => {
                                        // We check that the suggest type is actually
                                        // different from the received one
                                        // So we avoid suggestion method with Box<Self>
                                        // for instance
                                        self.tcx.at(span).type_of(*def_id).instantiate_identity()
                                            != rcvr_ty
                                    }
                                    (Mode::Path, false, _) => true,
                                    _ => false,
                                }
                            } else {
                                false
                            }
                        })
                        .collect::<Vec<_>>();
                    if !inherent_impls_candidate.is_empty() {
                        inherent_impls_candidate.sort_by_key(|id| self.tcx.def_path_str(id));
                        inherent_impls_candidate.dedup();

                        // number of types to show at most
                        let limit = if inherent_impls_candidate.len() == 5 { 5 } else { 4 };
                        let type_candidates = inherent_impls_candidate
                            .iter()
                            .take(limit)
                            .map(|impl_item| {
                                format!(
                                    "- `{}`",
                                    self.tcx.at(span).type_of(*impl_item).instantiate_identity()
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        let additional_types = if inherent_impls_candidate.len() > limit {
                            format!("\nand {} more types", inherent_impls_candidate.len() - limit)
                        } else {
                            "".to_string()
                        };
                        err.note(format!(
                            "the {item_kind} was found for\n{type_candidates}{additional_types}"
                        ));
                        find_candidate_for_method = mode == Mode::MethodCall;
                    }
                }
            } else {
                let ty_str =
                    if ty_str.len() > 50 { String::new() } else { format!("on `{ty_str}` ") };
                err.span_label(
                    span,
                    format!("{item_kind} cannot be called {ty_str}due to unsatisfied trait bounds"),
                );
            }
        };

        // If the method name is the name of a field with a function or closure type,
        // give a helping note that it has to be called as `(x.f)(...)`.
        if let SelfSource::MethodCall(expr) = source {
            if !self.suggest_calling_field_as_fn(span, rcvr_ty, expr, item_ident, &mut err)
                && similar_candidate.is_none()
                && !custom_span_label
            {
                label_span_not_found(&mut err);
            }
        } else if !custom_span_label {
            label_span_not_found(&mut err);
        }

        let confusable_suggested = self.confusable_method_name(
            &mut err,
            rcvr_ty,
            item_ident,
            args.map(|args| {
                args.iter()
                    .map(|expr| {
                        self.node_ty_opt(expr.hir_id).unwrap_or_else(|| self.next_ty_var(expr.span))
                    })
                    .collect()
            }),
        );

        // Don't suggest (for example) `expr.field.clone()` if `expr.clone()`
        // can't be called due to `typeof(expr): Clone` not holding.
        if unsatisfied_predicates.is_empty() {
            self.suggest_calling_method_on_field(
                &mut err,
                source,
                span,
                rcvr_ty,
                item_ident,
                expected.only_has_type(self),
            );
        }

        self.suggest_unwrapping_inner_self(&mut err, source, rcvr_ty, item_ident);

        for (span, mut bounds) in bound_spans {
            if !tcx.sess.source_map().is_span_accessible(span) {
                continue;
            }
            bounds.sort();
            bounds.dedup();
            let pre = if Some(span) == ty_span {
                ty_span.take();
                format!(
                    "{item_kind} `{item_ident}` not found for this {} because it ",
                    rcvr_ty.prefix_string(self.tcx)
                )
            } else {
                String::new()
            };
            let msg = match &bounds[..] {
                [bound] => format!("{pre}doesn't satisfy {bound}"),
                bounds if bounds.len() > 4 => format!("doesn't satisfy {} bounds", bounds.len()),
                [bounds @ .., last] => {
                    format!("{pre}doesn't satisfy {} or {last}", bounds.join(", "))
                }
                [] => unreachable!(),
            };
            err.span_label(span, msg);
        }
        if let Some(span) = ty_span {
            err.span_label(
                span,
                format!(
                    "{item_kind} `{item_ident}` not found for this {}",
                    rcvr_ty.prefix_string(self.tcx)
                ),
            );
        }

        if rcvr_ty.is_numeric() && rcvr_ty.is_fresh()
            || restrict_type_params
            || suggested_derive
            || self.lookup_alternative_tuple_impls(&mut err, &unsatisfied_predicates)
        {
        } else {
            self.suggest_traits_to_import(
                &mut err,
                span,
                rcvr_ty,
                item_ident,
                args.map(|args| args.len() + 1),
                source,
                no_match_data.out_of_scope_traits.clone(),
                static_candidates,
                unsatisfied_bounds,
                expected.only_has_type(self),
                trait_missing_method,
            );
        }

        // Don't emit a suggestion if we found an actual method
        // that had unsatisfied trait bounds
        if unsatisfied_predicates.is_empty() && rcvr_ty.is_enum() {
            let adt_def = rcvr_ty.ty_adt_def().expect("enum is not an ADT");
            if let Some(var_name) = edit_distance::find_best_match_for_name(
                &adt_def.variants().iter().map(|s| s.name).collect::<Vec<_>>(),
                item_ident.name,
                None,
            ) && let Some(variant) = adt_def.variants().iter().find(|s| s.name == var_name)
            {
                let mut suggestion = vec![(span, var_name.to_string())];
                if let SelfSource::QPath(ty) = source
                    && let hir::Node::Expr(ref path_expr) = self.tcx.parent_hir_node(ty.hir_id)
                    && let hir::ExprKind::Path(_) = path_expr.kind
                    && let hir::Node::Stmt(&hir::Stmt { kind: hir::StmtKind::Semi(parent), .. })
                    | hir::Node::Expr(parent) = self.tcx.parent_hir_node(path_expr.hir_id)
                {
                    let replacement_span =
                        if let hir::ExprKind::Call(..) | hir::ExprKind::Struct(..) = parent.kind {
                            // We want to replace the parts that need to go, like `()` and `{}`.
                            span.with_hi(parent.span.hi())
                        } else {
                            span
                        };
                    match (variant.ctor, parent.kind) {
                        (None, hir::ExprKind::Struct(..)) => {
                            // We want a struct and we have a struct. We won't suggest changing
                            // the fields (at least for now).
                            suggestion = vec![(span, var_name.to_string())];
                        }
                        (None, _) => {
                            // struct
                            suggestion = vec![(
                                replacement_span,
                                if variant.fields.is_empty() {
                                    format!("{var_name} {{}}")
                                } else {
                                    format!(
                                        "{var_name} {{ {} }}",
                                        variant
                                            .fields
                                            .iter()
                                            .map(|f| format!("{}: /* value */", f.name))
                                            .collect::<Vec<_>>()
                                            .join(", ")
                                    )
                                },
                            )];
                        }
                        (Some((hir::def::CtorKind::Const, _)), _) => {
                            // unit, remove the `()`.
                            suggestion = vec![(replacement_span, var_name.to_string())];
                        }
                        (
                            Some((hir::def::CtorKind::Fn, def_id)),
                            hir::ExprKind::Call(rcvr, args),
                        ) => {
                            let fn_sig = self.tcx.fn_sig(def_id).instantiate_identity();
                            let inputs = fn_sig.inputs().skip_binder();
                            // FIXME: reuse the logic for "change args" suggestion to account for types
                            // involved and detect things like substitution.
                            match (inputs, args) {
                                (inputs, []) => {
                                    // Add arguments.
                                    suggestion.push((
                                        rcvr.span.shrink_to_hi().with_hi(parent.span.hi()),
                                        format!(
                                            "({})",
                                            inputs
                                                .iter()
                                                .map(|i| format!("/* {i} */"))
                                                .collect::<Vec<String>>()
                                                .join(", ")
                                        ),
                                    ));
                                }
                                (_, [arg]) if inputs.len() != args.len() => {
                                    // Replace arguments.
                                    suggestion.push((
                                        arg.span,
                                        inputs
                                            .iter()
                                            .map(|i| format!("/* {i} */"))
                                            .collect::<Vec<String>>()
                                            .join(", "),
                                    ));
                                }
                                (_, [arg_start, .., arg_end]) if inputs.len() != args.len() => {
                                    // Replace arguments.
                                    suggestion.push((
                                        arg_start.span.to(arg_end.span),
                                        inputs
                                            .iter()
                                            .map(|i| format!("/* {i} */"))
                                            .collect::<Vec<String>>()
                                            .join(", "),
                                    ));
                                }
                                // Argument count is the same, keep as is.
                                _ => {}
                            }
                        }
                        (Some((hir::def::CtorKind::Fn, def_id)), _) => {
                            let fn_sig = self.tcx.fn_sig(def_id).instantiate_identity();
                            let inputs = fn_sig.inputs().skip_binder();
                            suggestion = vec![(
                                replacement_span,
                                format!(
                                    "{var_name}({})",
                                    inputs
                                        .iter()
                                        .map(|i| format!("/* {i} */"))
                                        .collect::<Vec<String>>()
                                        .join(", ")
                                ),
                            )];
                        }
                    }
                }
                err.multipart_suggestion_verbose(
                    "there is a variant with a similar name",
                    suggestion,
                    Applicability::HasPlaceholders,
                );
            }
        }

        if let Some(similar_candidate) = similar_candidate {
            // Don't emit a suggestion if we found an actual method
            // that had unsatisfied trait bounds
            if unsatisfied_predicates.is_empty()
                // ...or if we already suggested that name because of `rustc_confusable` annotation
                && Some(similar_candidate.name()) != confusable_suggested
                // and if the we aren't in an expansion.
                && !span.from_expansion()
            {
                self.find_likely_intended_associated_item(
                    &mut err,
                    similar_candidate,
                    span,
                    args,
                    mode,
                );
            }
        }

        if !find_candidate_for_method {
            self.lookup_segments_chain_for_no_match_method(
                &mut err,
                item_ident,
                item_kind,
                source,
                no_match_data,
            );
        }

        self.note_derefed_ty_has_method(&mut err, source, rcvr_ty, item_ident, expected);
        err.emit()
    }

    /// If the predicate failure is caused by an unmet bound on a tuple, recheck if the bound would
    /// succeed if all the types on the tuple had no borrows. This is a common problem for libraries
    /// like Bevy and ORMs, which rely heavily on traits being implemented on tuples.
    fn lookup_alternative_tuple_impls(
        &self,
        err: &mut Diag<'_>,
        unsatisfied_predicates: &[(
            ty::Predicate<'tcx>,
            Option<ty::Predicate<'tcx>>,
            Option<ObligationCause<'tcx>>,
        )],
    ) -> bool {
        let mut found_tuple = false;
        for (pred, root, _ob) in unsatisfied_predicates {
            let mut preds = vec![pred];
            if let Some(root) = root {
                // We will look at both the current predicate and the root predicate that caused it
                // to be needed. If calling something like `<(A, &B)>::default()`, then `pred` is
                // `&B: Default` and `root` is `(A, &B): Default`, which is the one we are checking
                // for further down, so we check both.
                preds.push(root);
            }
            for pred in preds {
                if let Some(clause) = pred.as_clause()
                    && let Some(clause) = clause.as_trait_clause()
                    && let ty = clause.self_ty().skip_binder()
                    && let ty::Tuple(types) = ty.kind()
                {
                    let path = clause.skip_binder().trait_ref.print_only_trait_path();
                    let def_id = clause.def_id();
                    let ty = Ty::new_tup(
                        self.tcx,
                        self.tcx.mk_type_list_from_iter(types.iter().map(|ty| ty.peel_refs())),
                    );
                    let args = ty::GenericArgs::for_item(self.tcx, def_id, |param, _| {
                        if param.index == 0 {
                            ty.into()
                        } else {
                            self.infcx.var_for_def(DUMMY_SP, param)
                        }
                    });
                    if self
                        .infcx
                        .type_implements_trait(def_id, args, self.param_env)
                        .must_apply_modulo_regions()
                    {
                        // "`Trait` is implemented for `(A, B)` but not for `(A, &B)`"
                        let mut msg = DiagStyledString::normal(format!("`{path}` "));
                        msg.push_highlighted("is");
                        msg.push_normal(" implemented for `(");
                        let len = types.len();
                        for (i, t) in types.iter().enumerate() {
                            msg.push(
                                format!("{}", with_forced_trimmed_paths!(t.peel_refs())),
                                t.peel_refs() != t,
                            );
                            if i < len - 1 {
                                msg.push_normal(", ");
                            }
                        }
                        msg.push_normal(")` but ");
                        msg.push_highlighted("not");
                        msg.push_normal(" for `(");
                        for (i, t) in types.iter().enumerate() {
                            msg.push(
                                format!("{}", with_forced_trimmed_paths!(t)),
                                t.peel_refs() != t,
                            );
                            if i < len - 1 {
                                msg.push_normal(", ");
                            }
                        }
                        msg.push_normal(")`");

                        // Find the span corresponding to the impl that was found to point at it.
                        if let Some(impl_span) = self
                            .tcx
                            .all_impls(def_id)
                            .filter(|&impl_def_id| {
                                let header = self.tcx.impl_trait_header(impl_def_id).unwrap();
                                let trait_ref = header.trait_ref.instantiate(
                                    self.tcx,
                                    self.infcx.fresh_args_for_item(DUMMY_SP, impl_def_id),
                                );

                                let value = ty::fold_regions(self.tcx, ty, |_, _| {
                                    self.tcx.lifetimes.re_erased
                                });
                                // FIXME: Don't bother dealing with non-lifetime binders here...
                                if value.has_escaping_bound_vars() {
                                    return false;
                                }
                                self.infcx.can_eq(ty::ParamEnv::empty(), trait_ref.self_ty(), value)
                                    && header.polarity == ty::ImplPolarity::Positive
                            })
                            .map(|impl_def_id| self.tcx.def_span(impl_def_id))
                            .next()
                        {
                            err.highlighted_span_note(impl_span, msg.0);
                        } else {
                            err.highlighted_note(msg.0);
                        }
                        found_tuple = true;
                    }
                    // If `pred` was already on the tuple, we don't need to look at the root
                    // obligation too.
                    break;
                }
            }
        }
        found_tuple
    }

    /// If an appropriate error source is not found, check method chain for possible candidates
    fn lookup_segments_chain_for_no_match_method(
        &self,
        err: &mut Diag<'_>,
        item_name: Ident,
        item_kind: &str,
        source: SelfSource<'tcx>,
        no_match_data: &NoMatchData<'tcx>,
    ) {
        if no_match_data.unsatisfied_predicates.is_empty()
            && let Mode::MethodCall = no_match_data.mode
            && let SelfSource::MethodCall(mut source_expr) = source
        {
            let mut stack_methods = vec![];
            while let hir::ExprKind::MethodCall(_path_segment, rcvr_expr, _args, method_span) =
                source_expr.kind
            {
                // Pop the matching receiver, to align on it's notional span
                if let Some(prev_match) = stack_methods.pop() {
                    err.span_label(
                        method_span,
                        format!("{item_kind} `{item_name}` is available on `{prev_match}`"),
                    );
                }
                let rcvr_ty = self.resolve_vars_if_possible(
                    self.typeck_results
                        .borrow()
                        .expr_ty_adjusted_opt(rcvr_expr)
                        .unwrap_or(Ty::new_misc_error(self.tcx)),
                );

                let Ok(candidates) = self.probe_for_name_many(
                    Mode::MethodCall,
                    item_name,
                    None,
                    IsSuggestion(true),
                    rcvr_ty,
                    source_expr.hir_id,
                    ProbeScope::TraitsInScope,
                ) else {
                    return;
                };

                // FIXME: `probe_for_name_many` searches for methods in inherent implementations,
                // so it may return a candidate that doesn't belong to this `revr_ty`. We need to
                // check whether the instantiated type matches the received one.
                for _matched_method in candidates {
                    // found a match, push to stack
                    stack_methods.push(rcvr_ty);
                }
                source_expr = rcvr_expr;
            }
            // If there is a match at the start of the chain, add a label for it too!
            if let Some(prev_match) = stack_methods.pop() {
                err.span_label(
                    source_expr.span,
                    format!("{item_kind} `{item_name}` is available on `{prev_match}`"),
                );
            }
        }
    }

    fn find_likely_intended_associated_item(
        &self,
        err: &mut Diag<'_>,
        similar_candidate: ty::AssocItem,
        span: Span,
        args: Option<&'tcx [hir::Expr<'tcx>]>,
        mode: Mode,
    ) {
        let tcx = self.tcx;
        let def_kind = similar_candidate.as_def_kind();
        let an = self.tcx.def_kind_descr_article(def_kind, similar_candidate.def_id);
        let similar_candidate_name = similar_candidate.name();
        let msg = format!(
            "there is {an} {} `{}` with a similar name",
            self.tcx.def_kind_descr(def_kind, similar_candidate.def_id),
            similar_candidate_name,
        );
        // Methods are defined within the context of a struct and their first parameter
        // is always `self`, which represents the instance of the struct the method is
        // being called on Associated functions don’t take self as a parameter and they are
        // not methods because they don’t have an instance of the struct to work with.
        if def_kind == DefKind::AssocFn {
            let ty_args = self.infcx.fresh_args_for_item(span, similar_candidate.def_id);
            let fn_sig = tcx.fn_sig(similar_candidate.def_id).instantiate(tcx, ty_args);
            let fn_sig = self.instantiate_binder_with_fresh_vars(span, infer::FnCall, fn_sig);
            if similar_candidate.is_method() {
                if let Some(args) = args
                    && fn_sig.inputs()[1..].len() == args.len()
                {
                    // We found a method with the same number of arguments as the method
                    // call expression the user wrote.
                    err.span_suggestion_verbose(
                        span,
                        msg,
                        similar_candidate_name,
                        Applicability::MaybeIncorrect,
                    );
                } else {
                    // We found a method but either the expression is not a method call or
                    // the argument count didn't match.
                    err.span_help(
                        tcx.def_span(similar_candidate.def_id),
                        format!(
                            "{msg}{}",
                            if let None = args { "" } else { ", but with different arguments" },
                        ),
                    );
                }
            } else if let Some(args) = args
                && fn_sig.inputs().len() == args.len()
            {
                // We have fn call expression and the argument count match the associated
                // function we found.
                err.span_suggestion_verbose(
                    span,
                    msg,
                    similar_candidate_name,
                    Applicability::MaybeIncorrect,
                );
            } else {
                err.span_help(tcx.def_span(similar_candidate.def_id), msg);
            }
        } else if let Mode::Path = mode
            && args.unwrap_or(&[]).is_empty()
        {
            // We have an associated item syntax and we found something that isn't an fn.
            err.span_suggestion_verbose(
                span,
                msg,
                similar_candidate_name,
                Applicability::MaybeIncorrect,
            );
        } else {
            // The expression is a function or method call, but the item we found is an
            // associated const or type.
            err.span_help(tcx.def_span(similar_candidate.def_id), msg);
        }
    }

    pub(crate) fn confusable_method_name(
        &self,
        err: &mut Diag<'_>,
        rcvr_ty: Ty<'tcx>,
        item_name: Ident,
        call_args: Option<Vec<Ty<'tcx>>>,
    ) -> Option<Symbol> {
        if let ty::Adt(adt, adt_args) = rcvr_ty.kind() {
            for inherent_impl_did in self.tcx.inherent_impls(adt.did()).into_iter() {
                for inherent_method in
                    self.tcx.associated_items(inherent_impl_did).in_definition_order()
                {
                    if let Some(candidates) = find_attr!(self.tcx.get_all_attrs(inherent_method.def_id), AttributeKind::Confusables{symbols, ..} => symbols)
                        && candidates.contains(&item_name.name)
                        && inherent_method.is_fn()
                    {
                        let args =
                            ty::GenericArgs::identity_for_item(self.tcx, inherent_method.def_id)
                                .rebase_onto(
                                    self.tcx,
                                    inherent_method.container_id(self.tcx),
                                    adt_args,
                                );
                        let fn_sig =
                            self.tcx.fn_sig(inherent_method.def_id).instantiate(self.tcx, args);
                        let fn_sig = self.instantiate_binder_with_fresh_vars(
                            item_name.span,
                            infer::FnCall,
                            fn_sig,
                        );
                        let name = inherent_method.name();
                        if let Some(ref args) = call_args
                            && fn_sig.inputs()[1..]
                                .iter()
                                .zip(args.into_iter())
                                .all(|(expected, found)| self.may_coerce(*expected, *found))
                            && fn_sig.inputs()[1..].len() == args.len()
                        {
                            err.span_suggestion_verbose(
                                item_name.span,
                                format!("you might have meant to use `{}`", name),
                                name,
                                Applicability::MaybeIncorrect,
                            );
                            return Some(name);
                        } else if let None = call_args {
                            err.span_note(
                                self.tcx.def_span(inherent_method.def_id),
                                format!("you might have meant to use method `{}`", name),
                            );
                            return Some(name);
                        }
                    }
                }
            }
        }
        None
    }
    fn note_candidates_on_method_error(
        &self,
        rcvr_ty: Ty<'tcx>,
        item_name: Ident,
        self_source: SelfSource<'tcx>,
        args: Option<&'tcx [hir::Expr<'tcx>]>,
        span: Span,
        err: &mut Diag<'_>,
        sources: &mut Vec<CandidateSource>,
        sugg_span: Option<Span>,
    ) {
        sources.sort_by_key(|source| match source {
            CandidateSource::Trait(id) => (0, self.tcx.def_path_str(id)),
            CandidateSource::Impl(id) => (1, self.tcx.def_path_str(id)),
        });
        sources.dedup();
        // Dynamic limit to avoid hiding just one candidate, which is silly.
        let limit = if sources.len() == 5 { 5 } else { 4 };

        let mut suggs = vec![];
        for (idx, source) in sources.iter().take(limit).enumerate() {
            match *source {
                CandidateSource::Impl(impl_did) => {
                    // Provide the best span we can. Use the item, if local to crate, else
                    // the impl, if local to crate (item may be defaulted), else nothing.
                    let Some(item) = self.associated_value(impl_did, item_name).or_else(|| {
                        let impl_trait_ref = self.tcx.impl_trait_ref(impl_did)?;
                        self.associated_value(impl_trait_ref.skip_binder().def_id, item_name)
                    }) else {
                        continue;
                    };

                    let note_span = if item.def_id.is_local() {
                        Some(self.tcx.def_span(item.def_id))
                    } else if impl_did.is_local() {
                        Some(self.tcx.def_span(impl_did))
                    } else {
                        None
                    };

                    let impl_ty = self.tcx.at(span).type_of(impl_did).instantiate_identity();

                    let insertion = match self.tcx.impl_trait_ref(impl_did) {
                        None => String::new(),
                        Some(trait_ref) => {
                            format!(
                                " of the trait `{}`",
                                self.tcx.def_path_str(trait_ref.skip_binder().def_id)
                            )
                        }
                    };

                    let (note_str, idx) = if sources.len() > 1 {
                        (
                            format!(
                                "candidate #{} is defined in an impl{} for the type `{}`",
                                idx + 1,
                                insertion,
                                impl_ty,
                            ),
                            Some(idx + 1),
                        )
                    } else {
                        (
                            format!(
                                "the candidate is defined in an impl{insertion} for the type `{impl_ty}`",
                            ),
                            None,
                        )
                    };
                    if let Some(note_span) = note_span {
                        // We have a span pointing to the method. Show note with snippet.
                        err.span_note(note_span, note_str);
                    } else {
                        err.note(note_str);
                    }
                    if let Some(sugg_span) = sugg_span
                        && let Some(trait_ref) = self.tcx.impl_trait_ref(impl_did)
                        && let Some(sugg) = print_disambiguation_help(
                            self.tcx,
                            err,
                            self_source,
                            args,
                            trait_ref
                                .instantiate(
                                    self.tcx,
                                    self.fresh_args_for_item(sugg_span, impl_did),
                                )
                                .with_self_ty(self.tcx, rcvr_ty),
                            idx,
                            sugg_span,
                            item,
                        )
                    {
                        suggs.push(sugg);
                    }
                }
                CandidateSource::Trait(trait_did) => {
                    let Some(item) = self.associated_value(trait_did, item_name) else { continue };
                    let item_span = self.tcx.def_span(item.def_id);
                    let idx = if sources.len() > 1 {
                        let msg = format!(
                            "candidate #{} is defined in the trait `{}`",
                            idx + 1,
                            self.tcx.def_path_str(trait_did)
                        );
                        err.span_note(item_span, msg);
                        Some(idx + 1)
                    } else {
                        let msg = format!(
                            "the candidate is defined in the trait `{}`",
                            self.tcx.def_path_str(trait_did)
                        );
                        err.span_note(item_span, msg);
                        None
                    };
                    if let Some(sugg_span) = sugg_span
                        && let Some(sugg) = print_disambiguation_help(
                            self.tcx,
                            err,
                            self_source,
                            args,
                            ty::TraitRef::new_from_args(
                                self.tcx,
                                trait_did,
                                self.fresh_args_for_item(sugg_span, trait_did),
                            )
                            .with_self_ty(self.tcx, rcvr_ty),
                            idx,
                            sugg_span,
                            item,
                        )
                    {
                        suggs.push(sugg);
                    }
                }
            }
        }
        if !suggs.is_empty()
            && let Some(span) = sugg_span
        {
            suggs.sort();
            err.span_suggestions(
                span.with_hi(item_name.span.lo()),
                "use fully-qualified syntax to disambiguate",
                suggs,
                Applicability::MachineApplicable,
            );
        }
        if sources.len() > limit {
            err.note(format!("and {} others", sources.len() - limit));
        }
    }

    /// Look at all the associated functions without receivers in the type's inherent impls
    /// to look for builders that return `Self`, `Option<Self>` or `Result<Self, _>`.
    fn find_builder_fn(&self, err: &mut Diag<'_>, rcvr_ty: Ty<'tcx>, expr_id: hir::HirId) {
        let ty::Adt(adt_def, _) = rcvr_ty.kind() else {
            return;
        };
        let mut items = self
            .tcx
            .inherent_impls(adt_def.did())
            .iter()
            .flat_map(|i| self.tcx.associated_items(i).in_definition_order())
            // Only assoc fn with no receivers and only if
            // they are resolvable
            .filter(|item| {
                matches!(item.kind, ty::AssocKind::Fn { has_self: false, .. })
                    && self
                        .probe_for_name(
                            Mode::Path,
                            item.ident(self.tcx),
                            None,
                            IsSuggestion(true),
                            rcvr_ty,
                            expr_id,
                            ProbeScope::TraitsInScope,
                        )
                        .is_ok()
            })
            .filter_map(|item| {
                // Only assoc fns that return `Self`, `Option<Self>` or `Result<Self, _>`.
                let ret_ty = self
                    .tcx
                    .fn_sig(item.def_id)
                    .instantiate(self.tcx, self.fresh_args_for_item(DUMMY_SP, item.def_id))
                    .output();
                let ret_ty = self.tcx.instantiate_bound_regions_with_erased(ret_ty);
                let ty::Adt(def, args) = ret_ty.kind() else {
                    return None;
                };
                // Check for `-> Self`
                if self.can_eq(self.param_env, ret_ty, rcvr_ty) {
                    return Some((item.def_id, ret_ty));
                }
                // Check for `-> Option<Self>` or `-> Result<Self, _>`
                if ![self.tcx.lang_items().option_type(), self.tcx.get_diagnostic_item(sym::Result)]
                    .contains(&Some(def.did()))
                {
                    return None;
                }
                let arg = args.get(0)?.expect_ty();
                if self.can_eq(self.param_env, rcvr_ty, arg) {
                    Some((item.def_id, ret_ty))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let post = if items.len() > 5 {
            let items_len = items.len();
            items.truncate(4);
            format!("\nand {} others", items_len - 4)
        } else {
            String::new()
        };
        match &items[..] {
            [] => {}
            [(def_id, ret_ty)] => {
                err.span_note(
                    self.tcx.def_span(def_id),
                    format!(
                        "if you're trying to build a new `{rcvr_ty}`, consider using `{}` which \
                         returns `{ret_ty}`",
                        self.tcx.def_path_str(def_id),
                    ),
                );
            }
            _ => {
                let span: MultiSpan = items
                    .iter()
                    .map(|(def_id, _)| self.tcx.def_span(def_id))
                    .collect::<Vec<Span>>()
                    .into();
                err.span_note(
                    span,
                    format!(
                        "if you're trying to build a new `{rcvr_ty}` consider using one of the \
                         following associated functions:\n{}{post}",
                        items
                            .iter()
                            .map(|(def_id, _ret_ty)| self.tcx.def_path_str(def_id))
                            .collect::<Vec<String>>()
                            .join("\n")
                    ),
                );
            }
        }
    }

    /// Suggest calling `Ty::method` if `.method()` isn't found because the method
    /// doesn't take a `self` receiver.
    fn suggest_associated_call_syntax(
        &self,
        err: &mut Diag<'_>,
        static_candidates: &Vec<CandidateSource>,
        rcvr_ty: Ty<'tcx>,
        source: SelfSource<'tcx>,
        item_name: Ident,
        args: Option<&'tcx [hir::Expr<'tcx>]>,
        sugg_span: Span,
    ) {
        let mut has_unsuggestable_args = false;
        let ty_str = if let Some(CandidateSource::Impl(impl_did)) = static_candidates.get(0) {
            // When the "method" is resolved through dereferencing, we really want the
            // original type that has the associated function for accurate suggestions.
            // (#61411)
            let impl_ty = self.tcx.type_of(*impl_did).instantiate_identity();
            let target_ty = self
                .autoderef(sugg_span, rcvr_ty)
                .silence_errors()
                .find(|(rcvr_ty, _)| {
                    DeepRejectCtxt::relate_rigid_infer(self.tcx).types_may_unify(*rcvr_ty, impl_ty)
                })
                .map_or(impl_ty, |(ty, _)| ty)
                .peel_refs();
            if let ty::Adt(def, args) = target_ty.kind() {
                // If there are any inferred arguments, (`{integer}`), we should replace
                // them with underscores to allow the compiler to infer them
                let infer_args = self.tcx.mk_args_from_iter(args.into_iter().map(|arg| {
                    if !arg.is_suggestable(self.tcx, true) {
                        has_unsuggestable_args = true;
                        match arg.kind() {
                            GenericArgKind::Lifetime(_) => self
                                .next_region_var(RegionVariableOrigin::MiscVariable(DUMMY_SP))
                                .into(),
                            GenericArgKind::Type(_) => self.next_ty_var(DUMMY_SP).into(),
                            GenericArgKind::Const(_) => self.next_const_var(DUMMY_SP).into(),
                        }
                    } else {
                        arg
                    }
                }));

                self.tcx.value_path_str_with_args(def.did(), infer_args)
            } else {
                self.ty_to_value_string(target_ty)
            }
        } else {
            self.ty_to_value_string(rcvr_ty.peel_refs())
        };
        if let SelfSource::MethodCall(_) = source {
            let first_arg = static_candidates.get(0).and_then(|candidate_source| {
                let (assoc_did, self_ty) = match candidate_source {
                    CandidateSource::Impl(impl_did) => {
                        (*impl_did, self.tcx.type_of(*impl_did).instantiate_identity())
                    }
                    CandidateSource::Trait(trait_did) => (*trait_did, rcvr_ty),
                };

                let assoc = self.associated_value(assoc_did, item_name)?;
                if !assoc.is_fn() {
                    return None;
                }

                // for CandidateSource::Impl, `Self` will be instantiated to a concrete type
                // but for CandidateSource::Trait, `Self` is still `Self`
                let sig = self.tcx.fn_sig(assoc.def_id).instantiate_identity();
                sig.inputs().skip_binder().get(0).and_then(|first| {
                    // if the type of first arg is the same as the current impl type, we should take the first arg into assoc function
                    let first_ty = first.peel_refs();
                    if first_ty == self_ty || first_ty == self.tcx.types.self_param {
                        Some(first.ref_mutability().map_or("", |mutbl| mutbl.ref_prefix_str()))
                    } else {
                        None
                    }
                })
            });

            let mut applicability = Applicability::MachineApplicable;
            let args = if let SelfSource::MethodCall(receiver) = source
                && let Some(args) = args
            {
                // The first arg is the same kind as the receiver
                let explicit_args = if first_arg.is_some() {
                    std::iter::once(receiver).chain(args.iter()).collect::<Vec<_>>()
                } else {
                    // There is no `Self` kind to infer the arguments from
                    if has_unsuggestable_args {
                        applicability = Applicability::HasPlaceholders;
                    }
                    args.iter().collect()
                };
                format!(
                    "({}{})",
                    first_arg.unwrap_or(""),
                    explicit_args
                        .iter()
                        .map(|arg| self
                            .tcx
                            .sess
                            .source_map()
                            .span_to_snippet(arg.span)
                            .unwrap_or_else(|_| {
                                applicability = Applicability::HasPlaceholders;
                                "_".to_owned()
                            }))
                        .collect::<Vec<_>>()
                        .join(", "),
                )
            } else {
                applicability = Applicability::HasPlaceholders;
                "(...)".to_owned()
            };
            err.span_suggestion(
                sugg_span,
                "use associated function syntax instead",
                format!("{ty_str}::{item_name}{args}"),
                applicability,
            );
        } else {
            err.help(format!("try with `{ty_str}::{item_name}`",));
        }
    }

    /// Suggest calling a field with a type that implements the `Fn*` traits instead of a method with
    /// the same name as the field i.e. `(a.my_fn_ptr)(10)` instead of `a.my_fn_ptr(10)`.
    fn suggest_calling_field_as_fn(
        &self,
        span: Span,
        rcvr_ty: Ty<'tcx>,
        expr: &hir::Expr<'_>,
        item_name: Ident,
        err: &mut Diag<'_>,
    ) -> bool {
        let tcx = self.tcx;
        let field_receiver =
            self.autoderef(span, rcvr_ty).silence_errors().find_map(|(ty, _)| match ty.kind() {
                ty::Adt(def, args) if !def.is_enum() => {
                    let variant = &def.non_enum_variant();
                    tcx.find_field_index(item_name, variant).map(|index| {
                        let field = &variant.fields[index];
                        let field_ty = field.ty(tcx, args);
                        (field, field_ty)
                    })
                }
                _ => None,
            });
        if let Some((field, field_ty)) = field_receiver {
            let scope = tcx.parent_module_from_def_id(self.body_id);
            let is_accessible = field.vis.is_accessible_from(scope, tcx);

            if is_accessible {
                if let Some((what, _, _)) = self.extract_callable_info(field_ty) {
                    let what = match what {
                        DefIdOrName::DefId(def_id) => self.tcx.def_descr(def_id),
                        DefIdOrName::Name(what) => what,
                    };
                    let expr_span = expr.span.to(item_name.span);
                    err.multipart_suggestion(
                        format!(
                            "to call the {what} stored in `{item_name}`, \
                            surround the field access with parentheses",
                        ),
                        vec![
                            (expr_span.shrink_to_lo(), '('.to_string()),
                            (expr_span.shrink_to_hi(), ')'.to_string()),
                        ],
                        Applicability::MachineApplicable,
                    );
                } else {
                    let call_expr = tcx.hir_expect_expr(tcx.parent_hir_id(expr.hir_id));

                    if let Some(span) = call_expr.span.trim_start(item_name.span) {
                        err.span_suggestion(
                            span,
                            "remove the arguments",
                            "",
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
            }

            let field_kind = if is_accessible { "field" } else { "private field" };
            err.span_label(item_name.span, format!("{field_kind}, not a method"));
            return true;
        }
        false
    }

    /// Suggest possible range with adding parentheses, for example:
    /// when encountering `0..1.map(|i| i + 1)` suggest `(0..1).map(|i| i + 1)`.
    fn report_failed_method_call_on_range_end(
        &self,
        tcx: TyCtxt<'tcx>,
        actual: Ty<'tcx>,
        source: SelfSource<'tcx>,
        span: Span,
        item_name: Ident,
        ty_str: &str,
        long_ty_path: &mut Option<PathBuf>,
    ) -> Result<(), ErrorGuaranteed> {
        if let SelfSource::MethodCall(expr) = source {
            for (_, parent) in tcx.hir_parent_iter(expr.hir_id).take(5) {
                if let Node::Expr(parent_expr) = parent {
                    let lang_item = match parent_expr.kind {
                        ExprKind::Struct(qpath, _, _) => match *qpath {
                            QPath::LangItem(LangItem::Range, ..) => Some(LangItem::Range),
                            QPath::LangItem(LangItem::RangeCopy, ..) => Some(LangItem::RangeCopy),
                            QPath::LangItem(LangItem::RangeInclusiveCopy, ..) => {
                                Some(LangItem::RangeInclusiveCopy)
                            }
                            QPath::LangItem(LangItem::RangeTo, ..) => Some(LangItem::RangeTo),
                            QPath::LangItem(LangItem::RangeToInclusive, ..) => {
                                Some(LangItem::RangeToInclusive)
                            }
                            _ => None,
                        },
                        ExprKind::Call(func, _) => match func.kind {
                            // `..=` desugars into `::std::ops::RangeInclusive::new(...)`.
                            ExprKind::Path(QPath::LangItem(LangItem::RangeInclusiveNew, ..)) => {
                                Some(LangItem::RangeInclusiveStruct)
                            }
                            _ => None,
                        },
                        _ => None,
                    };

                    if lang_item.is_none() {
                        continue;
                    }

                    let span_included = match parent_expr.kind {
                        hir::ExprKind::Struct(_, eps, _) => {
                            eps.len() > 0 && eps.last().is_some_and(|ep| ep.span.contains(span))
                        }
                        // `..=` desugars into `::std::ops::RangeInclusive::new(...)`.
                        hir::ExprKind::Call(func, ..) => func.span.contains(span),
                        _ => false,
                    };

                    if !span_included {
                        continue;
                    }

                    let Some(range_def_id) =
                        lang_item.and_then(|lang_item| self.tcx.lang_items().get(lang_item))
                    else {
                        continue;
                    };
                    let range_ty =
                        self.tcx.type_of(range_def_id).instantiate(self.tcx, &[actual.into()]);

                    let pick = self.lookup_probe_for_diagnostic(
                        item_name,
                        range_ty,
                        expr,
                        ProbeScope::AllTraits,
                        None,
                    );
                    if pick.is_ok() {
                        let range_span = parent_expr.span.with_hi(expr.span.hi());
                        let mut err = self.dcx().create_err(errors::MissingParenthesesInRange {
                            span,
                            ty_str: ty_str.to_string(),
                            method_name: item_name.as_str().to_string(),
                            add_missing_parentheses: Some(errors::AddMissingParenthesesInRange {
                                func_name: item_name.name.as_str().to_string(),
                                left: range_span.shrink_to_lo(),
                                right: range_span.shrink_to_hi(),
                            }),
                        });
                        *err.long_ty_path() = long_ty_path.take();
                        return Err(err.emit());
                    }
                }
            }
        }
        Ok(())
    }

    fn report_failed_method_call_on_numerical_infer_var(
        &self,
        tcx: TyCtxt<'tcx>,
        actual: Ty<'tcx>,
        source: SelfSource<'_>,
        span: Span,
        item_kind: &str,
        item_name: Ident,
        ty_str: &str,
        long_ty_path: &mut Option<PathBuf>,
    ) -> Result<(), ErrorGuaranteed> {
        let found_candidate = all_traits(self.tcx)
            .into_iter()
            .any(|info| self.associated_value(info.def_id, item_name).is_some());
        let found_assoc = |ty: Ty<'tcx>| {
            simplify_type(tcx, ty, TreatParams::InstantiateWithInfer)
                .and_then(|simp| {
                    tcx.incoherent_impls(simp)
                        .into_iter()
                        .find_map(|&id| self.associated_value(id, item_name))
                })
                .is_some()
        };
        let found_candidate = found_candidate
            || found_assoc(tcx.types.i8)
            || found_assoc(tcx.types.i16)
            || found_assoc(tcx.types.i32)
            || found_assoc(tcx.types.i64)
            || found_assoc(tcx.types.i128)
            || found_assoc(tcx.types.u8)
            || found_assoc(tcx.types.u16)
            || found_assoc(tcx.types.u32)
            || found_assoc(tcx.types.u64)
            || found_assoc(tcx.types.u128)
            || found_assoc(tcx.types.f32)
            || found_assoc(tcx.types.f64);
        if found_candidate
            && actual.is_numeric()
            && !actual.has_concrete_skeleton()
            && let SelfSource::MethodCall(expr) = source
        {
            let mut err = struct_span_code_err!(
                self.dcx(),
                span,
                E0689,
                "can't call {} `{}` on ambiguous numeric type `{}`",
                item_kind,
                item_name,
                ty_str
            );
            *err.long_ty_path() = long_ty_path.take();
            let concrete_type = if actual.is_integral() { "i32" } else { "f32" };
            match expr.kind {
                ExprKind::Lit(lit) => {
                    // numeric literal
                    let snippet = tcx
                        .sess
                        .source_map()
                        .span_to_snippet(lit.span)
                        .unwrap_or_else(|_| "<numeric literal>".to_owned());

                    // If this is a floating point literal that ends with '.',
                    // get rid of it to stop this from becoming a member access.
                    let snippet = snippet.strip_suffix('.').unwrap_or(&snippet);
                    err.span_suggestion(
                        lit.span,
                        format!(
                            "you must specify a concrete type for this numeric value, \
                                         like `{concrete_type}`"
                        ),
                        format!("{snippet}_{concrete_type}"),
                        Applicability::MaybeIncorrect,
                    );
                }
                ExprKind::Path(QPath::Resolved(_, path)) => {
                    // local binding
                    if let hir::def::Res::Local(hir_id) = path.res {
                        let span = tcx.hir_span(hir_id);
                        let filename = tcx.sess.source_map().span_to_filename(span);

                        let parent_node = self.tcx.parent_hir_node(hir_id);
                        let msg = format!(
                            "you must specify a type for this binding, like `{concrete_type}`",
                        );

                        match (filename, parent_node) {
                            (
                                FileName::Real(_),
                                Node::LetStmt(hir::LetStmt {
                                    source: hir::LocalSource::Normal,
                                    ty,
                                    ..
                                }),
                            ) => {
                                let type_span = ty
                                    .map(|ty| ty.span.with_lo(span.hi()))
                                    .unwrap_or(span.shrink_to_hi());
                                err.span_suggestion(
                                    // account for `let x: _ = 42;`
                                    //                   ^^^
                                    type_span,
                                    msg,
                                    format!(": {concrete_type}"),
                                    Applicability::MaybeIncorrect,
                                );
                            }
                            _ => {
                                err.span_label(span, msg);
                            }
                        }
                    }
                }
                _ => {}
            }
            return Err(err.emit());
        }
        Ok(())
    }

    /// For code `rect::area(...)`,
    /// if `rect` is a local variable and `area` is a valid assoc method for it,
    /// we try to suggest `rect.area()`
    pub(crate) fn suggest_assoc_method_call(&self, segs: &[PathSegment<'_>]) {
        debug!("suggest_assoc_method_call segs: {:?}", segs);
        let [seg1, seg2] = segs else {
            return;
        };
        self.dcx().try_steal_modify_and_emit_err(
            seg1.ident.span,
            StashKey::CallAssocMethod,
            |err| {
                let body = self.tcx.hir_body_owned_by(self.body_id);
                struct LetVisitor {
                    ident_name: Symbol,
                }

                // FIXME: This really should be taking scoping, etc into account.
                impl<'v> Visitor<'v> for LetVisitor {
                    type Result = ControlFlow<Option<&'v hir::Expr<'v>>>;
                    fn visit_stmt(&mut self, ex: &'v hir::Stmt<'v>) -> Self::Result {
                        if let hir::StmtKind::Let(&hir::LetStmt { pat, init, .. }) = ex.kind
                            && let hir::PatKind::Binding(_, _, ident, ..) = pat.kind
                            && ident.name == self.ident_name
                        {
                            ControlFlow::Break(init)
                        } else {
                            hir::intravisit::walk_stmt(self, ex)
                        }
                    }
                }

                if let Node::Expr(call_expr) = self.tcx.parent_hir_node(seg1.hir_id)
                    && let ControlFlow::Break(Some(expr)) =
                        (LetVisitor { ident_name: seg1.ident.name }).visit_body(&body)
                    && let Some(self_ty) = self.node_ty_opt(expr.hir_id)
                {
                    let probe = self.lookup_probe_for_diagnostic(
                        seg2.ident,
                        self_ty,
                        call_expr,
                        ProbeScope::TraitsInScope,
                        None,
                    );
                    if probe.is_ok() {
                        let sm = self.infcx.tcx.sess.source_map();
                        err.span_suggestion_verbose(
                            sm.span_extend_while(seg1.ident.span.shrink_to_hi(), |c| c == ':')
                                .unwrap(),
                            "you may have meant to call an instance method",
                            ".",
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
            },
        );
    }

    /// Suggest calling a method on a field i.e. `a.field.bar()` instead of `a.bar()`
    fn suggest_calling_method_on_field(
        &self,
        err: &mut Diag<'_>,
        source: SelfSource<'tcx>,
        span: Span,
        actual: Ty<'tcx>,
        item_name: Ident,
        return_type: Option<Ty<'tcx>>,
    ) {
        if let SelfSource::MethodCall(expr) = source {
            let mod_id = self.tcx.parent_module(expr.hir_id).to_def_id();
            for (fields, args) in self.get_field_candidates_considering_privacy_for_diag(
                span,
                actual,
                mod_id,
                expr.hir_id,
            ) {
                let call_expr = self.tcx.hir_expect_expr(self.tcx.parent_hir_id(expr.hir_id));

                let lang_items = self.tcx.lang_items();
                let never_mention_traits = [
                    lang_items.clone_trait(),
                    lang_items.deref_trait(),
                    lang_items.deref_mut_trait(),
                    self.tcx.get_diagnostic_item(sym::AsRef),
                    self.tcx.get_diagnostic_item(sym::AsMut),
                    self.tcx.get_diagnostic_item(sym::Borrow),
                    self.tcx.get_diagnostic_item(sym::BorrowMut),
                ];
                let mut candidate_fields: Vec<_> = fields
                    .into_iter()
                    .filter_map(|candidate_field| {
                        self.check_for_nested_field_satisfying_condition_for_diag(
                            span,
                            &|_, field_ty| {
                                self.lookup_probe_for_diagnostic(
                                    item_name,
                                    field_ty,
                                    call_expr,
                                    ProbeScope::TraitsInScope,
                                    return_type,
                                )
                                .is_ok_and(|pick| {
                                    !never_mention_traits
                                        .iter()
                                        .flatten()
                                        .any(|def_id| self.tcx.parent(pick.item.def_id) == *def_id)
                                })
                            },
                            candidate_field,
                            args,
                            vec![],
                            mod_id,
                            expr.hir_id,
                        )
                    })
                    .map(|field_path| {
                        field_path
                            .iter()
                            .map(|id| id.to_string())
                            .collect::<Vec<String>>()
                            .join(".")
                    })
                    .collect();
                candidate_fields.sort();

                let len = candidate_fields.len();
                if len > 0 {
                    err.span_suggestions(
                        item_name.span.shrink_to_lo(),
                        format!(
                            "{} of the expressions' fields {} a method of the same name",
                            if len > 1 { "some" } else { "one" },
                            if len > 1 { "have" } else { "has" },
                        ),
                        candidate_fields.iter().map(|path| format!("{path}.")),
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }
    }

    fn suggest_unwrapping_inner_self(
        &self,
        err: &mut Diag<'_>,
        source: SelfSource<'tcx>,
        actual: Ty<'tcx>,
        item_name: Ident,
    ) {
        let tcx = self.tcx;
        let SelfSource::MethodCall(expr) = source else {
            return;
        };
        let call_expr = tcx.hir_expect_expr(tcx.parent_hir_id(expr.hir_id));

        let ty::Adt(kind, args) = actual.kind() else {
            return;
        };
        match kind.adt_kind() {
            ty::AdtKind::Enum => {
                let matching_variants: Vec<_> = kind
                    .variants()
                    .iter()
                    .flat_map(|variant| {
                        let [field] = &variant.fields.raw[..] else {
                            return None;
                        };
                        let field_ty = field.ty(tcx, args);

                        // Skip `_`, since that'll just lead to ambiguity.
                        if self.resolve_vars_if_possible(field_ty).is_ty_var() {
                            return None;
                        }

                        self.lookup_probe_for_diagnostic(
                            item_name,
                            field_ty,
                            call_expr,
                            ProbeScope::TraitsInScope,
                            None,
                        )
                        .ok()
                        .map(|pick| (variant, field, pick))
                    })
                    .collect();

                let ret_ty_matches = |diagnostic_item| {
                    if let Some(ret_ty) = self
                        .ret_coercion
                        .as_ref()
                        .map(|c| self.resolve_vars_if_possible(c.borrow().expected_ty()))
                        && let ty::Adt(kind, _) = ret_ty.kind()
                        && tcx.get_diagnostic_item(diagnostic_item) == Some(kind.did())
                    {
                        true
                    } else {
                        false
                    }
                };

                match &matching_variants[..] {
                    [(_, field, pick)] => {
                        let self_ty = field.ty(tcx, args);
                        err.span_note(
                            tcx.def_span(pick.item.def_id),
                            format!("the method `{item_name}` exists on the type `{self_ty}`"),
                        );
                        let (article, kind, variant, question) =
                            if tcx.is_diagnostic_item(sym::Result, kind.did()) {
                                ("a", "Result", "Err", ret_ty_matches(sym::Result))
                            } else if tcx.is_diagnostic_item(sym::Option, kind.did()) {
                                ("an", "Option", "None", ret_ty_matches(sym::Option))
                            } else {
                                return;
                            };
                        if question {
                            err.span_suggestion_verbose(
                                expr.span.shrink_to_hi(),
                                format!(
                                    "use the `?` operator to extract the `{self_ty}` value, propagating \
                                    {article} `{kind}::{variant}` value to the caller"
                                ),
                                "?",
                                Applicability::MachineApplicable,
                            );
                        } else {
                            err.span_suggestion_verbose(
                                expr.span.shrink_to_hi(),
                                format!(
                                    "consider using `{kind}::expect` to unwrap the `{self_ty}` value, \
                                    panicking if the value is {article} `{kind}::{variant}`"
                                ),
                                ".expect(\"REASON\")",
                                Applicability::HasPlaceholders,
                            );
                        }
                    }
                    // FIXME(compiler-errors): Support suggestions for other matching enum variants
                    _ => {}
                }
            }
            // Target wrapper types - types that wrap or pretend to wrap another type,
            // perhaps this inner type is meant to be called?
            ty::AdtKind::Struct | ty::AdtKind::Union => {
                let [first] = ***args else {
                    return;
                };
                let ty::GenericArgKind::Type(ty) = first.kind() else {
                    return;
                };
                let Ok(pick) = self.lookup_probe_for_diagnostic(
                    item_name,
                    ty,
                    call_expr,
                    ProbeScope::TraitsInScope,
                    None,
                ) else {
                    return;
                };

                let name = self.ty_to_value_string(actual);
                let inner_id = kind.did();
                let mutable = if let Some(AutorefOrPtrAdjustment::Autoref { mutbl, .. }) =
                    pick.autoref_or_ptr_adjustment
                {
                    Some(mutbl)
                } else {
                    None
                };

                if tcx.is_diagnostic_item(sym::LocalKey, inner_id) {
                    err.help("use `with` or `try_with` to access thread local storage");
                } else if tcx.is_lang_item(kind.did(), LangItem::MaybeUninit) {
                    err.help(format!(
                        "if this `{name}` has been initialized, \
                        use one of the `assume_init` methods to access the inner value"
                    ));
                } else if tcx.is_diagnostic_item(sym::RefCell, inner_id) {
                    let (suggestion, borrow_kind, panic_if) = match mutable {
                        Some(Mutability::Not) => (".borrow()", "borrow", "a mutable borrow exists"),
                        Some(Mutability::Mut) => {
                            (".borrow_mut()", "mutably borrow", "any borrows exist")
                        }
                        None => return,
                    };
                    err.span_suggestion_verbose(
                        expr.span.shrink_to_hi(),
                        format!(
                            "use `{suggestion}` to {borrow_kind} the `{ty}`, \
                            panicking if {panic_if}"
                        ),
                        suggestion,
                        Applicability::MaybeIncorrect,
                    );
                } else if tcx.is_diagnostic_item(sym::Mutex, inner_id) {
                    err.span_suggestion_verbose(
                        expr.span.shrink_to_hi(),
                        format!(
                            "use `.lock().unwrap()` to borrow the `{ty}`, \
                            blocking the current thread until it can be acquired"
                        ),
                        ".lock().unwrap()",
                        Applicability::MaybeIncorrect,
                    );
                } else if tcx.is_diagnostic_item(sym::RwLock, inner_id) {
                    let (suggestion, borrow_kind) = match mutable {
                        Some(Mutability::Not) => (".read().unwrap()", "borrow"),
                        Some(Mutability::Mut) => (".write().unwrap()", "mutably borrow"),
                        None => return,
                    };
                    err.span_suggestion_verbose(
                        expr.span.shrink_to_hi(),
                        format!(
                            "use `{suggestion}` to {borrow_kind} the `{ty}`, \
                            blocking the current thread until it can be acquired"
                        ),
                        suggestion,
                        Applicability::MaybeIncorrect,
                    );
                } else {
                    return;
                };

                err.span_note(
                    tcx.def_span(pick.item.def_id),
                    format!("the method `{item_name}` exists on the type `{ty}`"),
                );
            }
        }
    }

    pub(crate) fn note_unmet_impls_on_type(
        &self,
        err: &mut Diag<'_>,
        errors: Vec<FulfillmentError<'tcx>>,
        suggest_derive: bool,
    ) {
        let preds: Vec<_> = errors
            .iter()
            .filter_map(|e| match e.obligation.predicate.kind().skip_binder() {
                ty::PredicateKind::Clause(ty::ClauseKind::Trait(pred)) => {
                    match pred.self_ty().kind() {
                        ty::Adt(_, _) => Some(pred),
                        _ => None,
                    }
                }
                _ => None,
            })
            .collect();

        // Note for local items and foreign items respectively.
        let (mut local_preds, mut foreign_preds): (Vec<_>, Vec<_>) =
            preds.iter().partition(|&pred| {
                if let ty::Adt(def, _) = pred.self_ty().kind() {
                    def.did().is_local()
                } else {
                    false
                }
            });

        local_preds.sort_by_key(|pred: &&ty::TraitPredicate<'_>| pred.trait_ref.to_string());
        let local_def_ids = local_preds
            .iter()
            .filter_map(|pred| match pred.self_ty().kind() {
                ty::Adt(def, _) => Some(def.did()),
                _ => None,
            })
            .collect::<FxIndexSet<_>>();
        let mut local_spans: MultiSpan = local_def_ids
            .iter()
            .filter_map(|def_id| {
                let span = self.tcx.def_span(*def_id);
                if span.is_dummy() { None } else { Some(span) }
            })
            .collect::<Vec<_>>()
            .into();
        for pred in &local_preds {
            match pred.self_ty().kind() {
                ty::Adt(def, _) => {
                    local_spans.push_span_label(
                        self.tcx.def_span(def.did()),
                        format!("must implement `{}`", pred.trait_ref.print_trait_sugared()),
                    );
                }
                _ => {}
            }
        }
        if local_spans.primary_span().is_some() {
            let msg = if let [local_pred] = local_preds.as_slice() {
                format!(
                    "an implementation of `{}` might be missing for `{}`",
                    local_pred.trait_ref.print_trait_sugared(),
                    local_pred.self_ty()
                )
            } else {
                format!(
                    "the following type{} would have to `impl` {} required trait{} for this \
                     operation to be valid",
                    pluralize!(local_def_ids.len()),
                    if local_def_ids.len() == 1 { "its" } else { "their" },
                    pluralize!(local_preds.len()),
                )
            };
            err.span_note(local_spans, msg);
        }

        foreign_preds.sort_by_key(|pred: &&ty::TraitPredicate<'_>| pred.trait_ref.to_string());
        let foreign_def_ids = foreign_preds
            .iter()
            .filter_map(|pred| match pred.self_ty().kind() {
                ty::Adt(def, _) => Some(def.did()),
                _ => None,
            })
            .collect::<FxIndexSet<_>>();
        let mut foreign_spans: MultiSpan = foreign_def_ids
            .iter()
            .filter_map(|def_id| {
                let span = self.tcx.def_span(*def_id);
                if span.is_dummy() { None } else { Some(span) }
            })
            .collect::<Vec<_>>()
            .into();
        for pred in &foreign_preds {
            match pred.self_ty().kind() {
                ty::Adt(def, _) => {
                    foreign_spans.push_span_label(
                        self.tcx.def_span(def.did()),
                        format!("not implement `{}`", pred.trait_ref.print_trait_sugared()),
                    );
                }
                _ => {}
            }
        }
        if foreign_spans.primary_span().is_some() {
            let msg = if let [foreign_pred] = foreign_preds.as_slice() {
                format!(
                    "the foreign item type `{}` doesn't implement `{}`",
                    foreign_pred.self_ty(),
                    foreign_pred.trait_ref.print_trait_sugared()
                )
            } else {
                format!(
                    "the foreign item type{} {} implement required trait{} for this \
                     operation to be valid",
                    pluralize!(foreign_def_ids.len()),
                    if foreign_def_ids.len() > 1 { "don't" } else { "doesn't" },
                    pluralize!(foreign_preds.len()),
                )
            };
            err.span_note(foreign_spans, msg);
        }

        let preds: Vec<_> = errors
            .iter()
            .map(|e| (e.obligation.predicate, None, Some(e.obligation.cause.clone())))
            .collect();
        if suggest_derive {
            self.suggest_derive(err, &preds);
        } else {
            // The predicate comes from a binop where the lhs and rhs have different types.
            let _ = self.note_predicate_source_and_get_derives(err, &preds);
        }
    }

    fn note_predicate_source_and_get_derives(
        &self,
        err: &mut Diag<'_>,
        unsatisfied_predicates: &[(
            ty::Predicate<'tcx>,
            Option<ty::Predicate<'tcx>>,
            Option<ObligationCause<'tcx>>,
        )],
    ) -> Vec<(String, Span, Symbol)> {
        let mut derives = Vec::<(String, Span, Symbol)>::new();
        let mut traits = Vec::new();
        for (pred, _, _) in unsatisfied_predicates {
            let Some(ty::PredicateKind::Clause(ty::ClauseKind::Trait(trait_pred))) =
                pred.kind().no_bound_vars()
            else {
                continue;
            };
            let adt = match trait_pred.self_ty().ty_adt_def() {
                Some(adt) if adt.did().is_local() => adt,
                _ => continue,
            };
            if let Some(diagnostic_name) = self.tcx.get_diagnostic_name(trait_pred.def_id()) {
                let can_derive = match diagnostic_name {
                    sym::Default => !adt.is_enum(),
                    sym::Eq
                    | sym::PartialEq
                    | sym::Ord
                    | sym::PartialOrd
                    | sym::Clone
                    | sym::Copy
                    | sym::Hash
                    | sym::Debug => true,
                    _ => false,
                };
                if can_derive {
                    let self_name = trait_pred.self_ty().to_string();
                    let self_span = self.tcx.def_span(adt.did());
                    for super_trait in
                        supertraits(self.tcx, ty::Binder::dummy(trait_pred.trait_ref))
                    {
                        if let Some(parent_diagnostic_name) =
                            self.tcx.get_diagnostic_name(super_trait.def_id())
                        {
                            derives.push((self_name.clone(), self_span, parent_diagnostic_name));
                        }
                    }
                    derives.push((self_name, self_span, diagnostic_name));
                } else {
                    traits.push(trait_pred.def_id());
                }
            } else {
                traits.push(trait_pred.def_id());
            }
        }
        traits.sort_by_key(|id| self.tcx.def_path_str(id));
        traits.dedup();

        let len = traits.len();
        if len > 0 {
            let span =
                MultiSpan::from_spans(traits.iter().map(|&did| self.tcx.def_span(did)).collect());
            let mut names = format!("`{}`", self.tcx.def_path_str(traits[0]));
            for (i, &did) in traits.iter().enumerate().skip(1) {
                if len > 2 {
                    names.push_str(", ");
                }
                if i == len - 1 {
                    names.push_str(" and ");
                }
                names.push('`');
                names.push_str(&self.tcx.def_path_str(did));
                names.push('`');
            }
            err.span_note(
                span,
                format!("the trait{} {} must be implemented", pluralize!(len), names),
            );
        }

        derives
    }

    pub(crate) fn suggest_derive(
        &self,
        err: &mut Diag<'_>,
        unsatisfied_predicates: &[(
            ty::Predicate<'tcx>,
            Option<ty::Predicate<'tcx>>,
            Option<ObligationCause<'tcx>>,
        )],
    ) -> bool {
        let mut derives = self.note_predicate_source_and_get_derives(err, unsatisfied_predicates);
        derives.sort();
        derives.dedup();

        let mut derives_grouped = Vec::<(String, Span, String)>::new();
        for (self_name, self_span, trait_name) in derives.into_iter() {
            if let Some((last_self_name, _, last_trait_names)) = derives_grouped.last_mut() {
                if last_self_name == &self_name {
                    last_trait_names.push_str(format!(", {trait_name}").as_str());
                    continue;
                }
            }
            derives_grouped.push((self_name, self_span, trait_name.to_string()));
        }

        for (self_name, self_span, traits) in &derives_grouped {
            err.span_suggestion_verbose(
                self_span.shrink_to_lo(),
                format!("consider annotating `{self_name}` with `#[derive({traits})]`"),
                format!("#[derive({traits})]\n"),
                Applicability::MaybeIncorrect,
            );
        }
        !derives_grouped.is_empty()
    }

    fn note_derefed_ty_has_method(
        &self,
        err: &mut Diag<'_>,
        self_source: SelfSource<'tcx>,
        rcvr_ty: Ty<'tcx>,
        item_name: Ident,
        expected: Expectation<'tcx>,
    ) {
        let SelfSource::QPath(ty) = self_source else {
            return;
        };
        for (deref_ty, _) in self.autoderef(DUMMY_SP, rcvr_ty).silence_errors().skip(1) {
            if let Ok(pick) = self.probe_for_name(
                Mode::Path,
                item_name,
                expected.only_has_type(self),
                IsSuggestion(true),
                deref_ty,
                ty.hir_id,
                ProbeScope::TraitsInScope,
            ) {
                if deref_ty.is_suggestable(self.tcx, true)
                    // If this method receives `&self`, then the provided
                    // argument _should_ coerce, so it's valid to suggest
                    // just changing the path.
                    && pick.item.is_method()
                    && let Some(self_ty) =
                        self.tcx.fn_sig(pick.item.def_id).instantiate_identity().inputs().skip_binder().get(0)
                    && self_ty.is_ref()
                {
                    let suggested_path = match deref_ty.kind() {
                        ty::Bool
                        | ty::Char
                        | ty::Int(_)
                        | ty::Uint(_)
                        | ty::Float(_)
                        | ty::Adt(_, _)
                        | ty::Str
                        | ty::Alias(ty::Projection | ty::Inherent, _)
                        | ty::Param(_) => format!("{deref_ty}"),
                        // we need to test something like  <&[_]>::len or <(&[u32])>::len
                        // and Vec::function();
                        // <&[_]>::len or <&[u32]>::len doesn't need an extra "<>" between
                        // but for Adt type like Vec::function()
                        // we would suggest <[_]>::function();
                        _ if self
                            .tcx
                            .sess
                            .source_map()
                            .span_wrapped_by_angle_or_parentheses(ty.span) =>
                        {
                            format!("{deref_ty}")
                        }
                        _ => format!("<{deref_ty}>"),
                    };
                    err.span_suggestion_verbose(
                        ty.span,
                        format!("the function `{item_name}` is implemented on `{deref_ty}`"),
                        suggested_path,
                        Applicability::MaybeIncorrect,
                    );
                } else {
                    err.span_note(
                        ty.span,
                        format!("the function `{item_name}` is implemented on `{deref_ty}`"),
                    );
                }
                return;
            }
        }
    }

    /// Print out the type for use in value namespace.
    fn ty_to_value_string(&self, ty: Ty<'tcx>) -> String {
        match ty.kind() {
            ty::Adt(def, args) => self.tcx.def_path_str_with_args(def.did(), args),
            _ => self.ty_to_string(ty),
        }
    }

    fn suggest_await_before_method(
        &self,
        err: &mut Diag<'_>,
        item_name: Ident,
        ty: Ty<'tcx>,
        call: &hir::Expr<'_>,
        span: Span,
        return_type: Option<Ty<'tcx>>,
    ) {
        let output_ty = match self.err_ctxt().get_impl_future_output_ty(ty) {
            Some(output_ty) => self.resolve_vars_if_possible(output_ty),
            _ => return,
        };
        let method_exists =
            self.method_exists_for_diagnostic(item_name, output_ty, call.hir_id, return_type);
        debug!("suggest_await_before_method: is_method_exist={}", method_exists);
        if method_exists {
            err.span_suggestion_verbose(
                span.shrink_to_lo(),
                "consider `await`ing on the `Future` and calling the method on its `Output`",
                "await.",
                Applicability::MaybeIncorrect,
            );
        }
    }

    fn suggest_use_candidates<F>(&self, candidates: Vec<DefId>, handle_candidates: F)
    where
        F: FnOnce(Vec<String>, Vec<String>, Span),
    {
        let parent_map = self.tcx.visible_parent_map(());

        let scope = self.tcx.parent_module_from_def_id(self.body_id);
        let (accessible_candidates, inaccessible_candidates): (Vec<_>, Vec<_>) =
            candidates.into_iter().partition(|id| {
                let vis = self.tcx.visibility(*id);
                vis.is_accessible_from(scope, self.tcx)
            });

        let sugg = |candidates: Vec<_>, visible| {
            // Separate out candidates that must be imported with a glob, because they are named `_`
            // and cannot be referred with their identifier.
            let (candidates, globs): (Vec<_>, Vec<_>) =
                candidates.into_iter().partition(|trait_did| {
                    if let Some(parent_did) = parent_map.get(trait_did) {
                        // If the item is re-exported as `_`, we should suggest a glob-import instead.
                        if *parent_did != self.tcx.parent(*trait_did)
                            && self
                                .tcx
                                .module_children(*parent_did)
                                .iter()
                                .filter(|child| child.res.opt_def_id() == Some(*trait_did))
                                .all(|child| child.ident.name == kw::Underscore)
                        {
                            return false;
                        }
                    }

                    true
                });

            let prefix = if visible { "use " } else { "" };
            let postfix = if visible { ";" } else { "" };
            let path_strings = candidates.iter().map(|trait_did| {
                format!(
                    "{prefix}{}{postfix}\n",
                    with_no_visible_paths_if_doc_hidden!(with_crate_prefix!(
                        self.tcx.def_path_str(*trait_did)
                    )),
                )
            });

            let glob_path_strings = globs.iter().map(|trait_did| {
                let parent_did = parent_map.get(trait_did).unwrap();
                format!(
                    "{prefix}{}::*{postfix} // trait {}\n",
                    with_no_visible_paths_if_doc_hidden!(with_crate_prefix!(
                        self.tcx.def_path_str(*parent_did)
                    )),
                    self.tcx.item_name(*trait_did),
                )
            });
            let mut sugg: Vec<_> = path_strings.chain(glob_path_strings).collect();
            sugg.sort();
            sugg
        };

        let accessible_sugg = sugg(accessible_candidates, true);
        let inaccessible_sugg = sugg(inaccessible_candidates, false);

        let (module, _, _) = self.tcx.hir_get_module(scope);
        let span = module.spans.inject_use_span;
        handle_candidates(accessible_sugg, inaccessible_sugg, span);
    }

    fn suggest_valid_traits(
        &self,
        err: &mut Diag<'_>,
        item_name: Ident,
        valid_out_of_scope_traits: Vec<DefId>,
        explain: bool,
    ) -> bool {
        if !valid_out_of_scope_traits.is_empty() {
            let mut candidates = valid_out_of_scope_traits;
            candidates.sort_by_key(|id| self.tcx.def_path_str(id));
            candidates.dedup();

            // `TryFrom` and `FromIterator` have no methods
            let edition_fix = candidates
                .iter()
                .find(|did| self.tcx.is_diagnostic_item(sym::TryInto, **did))
                .copied();

            if explain {
                err.help("items from traits can only be used if the trait is in scope");
            }

            let msg = format!(
                "{this_trait_is} implemented but not in scope",
                this_trait_is = if candidates.len() == 1 {
                    format!(
                        "trait `{}` which provides `{item_name}` is",
                        self.tcx.item_name(candidates[0]),
                    )
                } else {
                    format!("the following traits which provide `{item_name}` are")
                }
            );

            self.suggest_use_candidates(candidates, |accessible_sugg, inaccessible_sugg, span| {
                let suggest_for_access = |err: &mut Diag<'_>, mut msg: String, suggs: Vec<_>| {
                    msg += &format!(
                        "; perhaps you want to import {one_of}",
                        one_of = if suggs.len() == 1 { "it" } else { "one of them" },
                    );
                    err.span_suggestions(span, msg, suggs, Applicability::MaybeIncorrect);
                };
                let suggest_for_privacy = |err: &mut Diag<'_>, suggs: Vec<String>| {
                    let msg = format!(
                        "{this_trait_is} implemented but not reachable",
                        this_trait_is = if let [sugg] = suggs.as_slice() {
                            format!("trait `{}` which provides `{item_name}` is", sugg.trim())
                        } else {
                            format!("the following traits which provide `{item_name}` are")
                        }
                    );
                    if suggs.len() == 1 {
                        err.help(msg);
                    } else {
                        err.span_suggestions(span, msg, suggs, Applicability::MaybeIncorrect);
                    }
                };
                if accessible_sugg.is_empty() {
                    // `inaccessible_sugg` must not be empty
                    suggest_for_privacy(err, inaccessible_sugg);
                } else if inaccessible_sugg.is_empty() {
                    suggest_for_access(err, msg, accessible_sugg);
                } else {
                    suggest_for_access(err, msg, accessible_sugg);
                    suggest_for_privacy(err, inaccessible_sugg);
                }
            });

            if let Some(did) = edition_fix {
                err.note(format!(
                    "'{}' is included in the prelude starting in Edition 2021",
                    with_crate_prefix!(self.tcx.def_path_str(did))
                ));
            }

            true
        } else {
            false
        }
    }

    fn suggest_traits_to_import(
        &self,
        err: &mut Diag<'_>,
        span: Span,
        rcvr_ty: Ty<'tcx>,
        item_name: Ident,
        inputs_len: Option<usize>,
        source: SelfSource<'tcx>,
        valid_out_of_scope_traits: Vec<DefId>,
        static_candidates: &[CandidateSource],
        unsatisfied_bounds: bool,
        return_type: Option<Ty<'tcx>>,
        trait_missing_method: bool,
    ) {
        let mut alt_rcvr_sugg = false;
        let mut trait_in_other_version_found = false;
        if let (SelfSource::MethodCall(rcvr), false) = (source, unsatisfied_bounds) {
            debug!(
                "suggest_traits_to_import: span={:?}, item_name={:?}, rcvr_ty={:?}, rcvr={:?}",
                span, item_name, rcvr_ty, rcvr
            );
            let skippable = [
                self.tcx.lang_items().clone_trait(),
                self.tcx.lang_items().deref_trait(),
                self.tcx.lang_items().deref_mut_trait(),
                self.tcx.lang_items().drop_trait(),
                self.tcx.get_diagnostic_item(sym::AsRef),
            ];
            // Try alternative arbitrary self types that could fulfill this call.
            // FIXME: probe for all types that *could* be arbitrary self-types, not
            // just this list.
            for (rcvr_ty, post, pin_call) in &[
                (rcvr_ty, "", None),
                (
                    Ty::new_mut_ref(self.tcx, self.tcx.lifetimes.re_erased, rcvr_ty),
                    "&mut ",
                    Some("as_mut"),
                ),
                (
                    Ty::new_imm_ref(self.tcx, self.tcx.lifetimes.re_erased, rcvr_ty),
                    "&",
                    Some("as_ref"),
                ),
            ] {
                match self.lookup_probe_for_diagnostic(
                    item_name,
                    *rcvr_ty,
                    rcvr,
                    ProbeScope::AllTraits,
                    return_type,
                ) {
                    Ok(pick) => {
                        // If the method is defined for the receiver we have, it likely wasn't `use`d.
                        // We point at the method, but we just skip the rest of the check for arbitrary
                        // self types and rely on the suggestion to `use` the trait from
                        // `suggest_valid_traits`.
                        let did = Some(pick.item.container_id(self.tcx));
                        if skippable.contains(&did) {
                            continue;
                        }
                        trait_in_other_version_found = self
                            .detect_and_explain_multiple_crate_versions_of_trait_item(
                                err,
                                pick.item.def_id,
                                rcvr.hir_id,
                                Some(*rcvr_ty),
                            );
                        if pick.autoderefs == 0 && !trait_in_other_version_found {
                            err.span_label(
                                pick.item.ident(self.tcx).span,
                                format!("the method is available for `{rcvr_ty}` here"),
                            );
                        }
                        break;
                    }
                    Err(MethodError::Ambiguity(_)) => {
                        // If the method is defined (but ambiguous) for the receiver we have, it is also
                        // likely we haven't `use`d it. It may be possible that if we `Box`/`Pin`/etc.
                        // the receiver, then it might disambiguate this method, but I think these
                        // suggestions are generally misleading (see #94218).
                        break;
                    }
                    Err(_) => (),
                }

                let Some(unpin_trait) = self.tcx.lang_items().unpin_trait() else {
                    return;
                };
                let pred = ty::TraitRef::new(self.tcx, unpin_trait, [*rcvr_ty]);
                let unpin = self.predicate_must_hold_considering_regions(&Obligation::new(
                    self.tcx,
                    self.misc(rcvr.span),
                    self.param_env,
                    pred,
                ));
                for (rcvr_ty, pre) in &[
                    (Ty::new_lang_item(self.tcx, *rcvr_ty, LangItem::OwnedBox), "Box::new"),
                    (Ty::new_lang_item(self.tcx, *rcvr_ty, LangItem::Pin), "Pin::new"),
                    (Ty::new_diagnostic_item(self.tcx, *rcvr_ty, sym::Arc), "Arc::new"),
                    (Ty::new_diagnostic_item(self.tcx, *rcvr_ty, sym::Rc), "Rc::new"),
                ] {
                    if let Some(new_rcvr_t) = *rcvr_ty
                        && let Ok(pick) = self.lookup_probe_for_diagnostic(
                            item_name,
                            new_rcvr_t,
                            rcvr,
                            ProbeScope::AllTraits,
                            return_type,
                        )
                    {
                        debug!("try_alt_rcvr: pick candidate {:?}", pick);
                        let did = Some(pick.item.container_id(self.tcx));
                        // We don't want to suggest a container type when the missing
                        // method is `.clone()` or `.deref()` otherwise we'd suggest
                        // `Arc::new(foo).clone()`, which is far from what the user wants.
                        // Explicitly ignore the `Pin::as_ref()` method as `Pin` does not
                        // implement the `AsRef` trait.
                        let skip = skippable.contains(&did)
                            || (("Pin::new" == *pre)
                                && ((sym::as_ref == item_name.name) || !unpin))
                            || inputs_len.is_some_and(|inputs_len| {
                                pick.item.is_fn()
                                    && self
                                        .tcx
                                        .fn_sig(pick.item.def_id)
                                        .skip_binder()
                                        .skip_binder()
                                        .inputs()
                                        .len()
                                        != inputs_len
                            });
                        // Make sure the method is defined for the *actual* receiver: we don't
                        // want to treat `Box<Self>` as a receiver if it only works because of
                        // an autoderef to `&self`
                        if pick.autoderefs == 0 && !skip {
                            err.span_label(
                                pick.item.ident(self.tcx).span,
                                format!("the method is available for `{new_rcvr_t}` here"),
                            );
                            err.multipart_suggestion(
                                "consider wrapping the receiver expression with the \
                                 appropriate type",
                                vec![
                                    (rcvr.span.shrink_to_lo(), format!("{pre}({post}")),
                                    (rcvr.span.shrink_to_hi(), ")".to_string()),
                                ],
                                Applicability::MaybeIncorrect,
                            );
                            // We don't care about the other suggestions.
                            alt_rcvr_sugg = true;
                        }
                    }
                }
                // We special case the situation where `Pin::new` wouldn't work, and instead
                // suggest using the `pin!()` macro instead.
                if let Some(new_rcvr_t) = Ty::new_lang_item(self.tcx, *rcvr_ty, LangItem::Pin)
                    // We didn't find an alternative receiver for the method.
                    && !alt_rcvr_sugg
                    // `T: !Unpin`
                    && !unpin
                    // The method isn't `as_ref`, as it would provide a wrong suggestion for `Pin`.
                    && sym::as_ref != item_name.name
                    // Either `Pin::as_ref` or `Pin::as_mut`.
                    && let Some(pin_call) = pin_call
                    // Search for `item_name` as a method accessible on `Pin<T>`.
                    && let Ok(pick) = self.lookup_probe_for_diagnostic(
                        item_name,
                        new_rcvr_t,
                        rcvr,
                        ProbeScope::AllTraits,
                        return_type,
                    )
                    // We skip some common traits that we don't want to consider because autoderefs
                    // would take care of them.
                    && !skippable.contains(&Some(pick.item.container_id(self.tcx)))
                    // We don't want to go through derefs.
                    && pick.autoderefs == 0
                    // Check that the method of the same name that was found on the new `Pin<T>`
                    // receiver has the same number of arguments that appear in the user's code.
                    && inputs_len.is_some_and(|inputs_len| pick.item.is_fn() && self.tcx.fn_sig(pick.item.def_id).skip_binder().skip_binder().inputs().len() == inputs_len)
                {
                    let indent = self
                        .tcx
                        .sess
                        .source_map()
                        .indentation_before(rcvr.span)
                        .unwrap_or_else(|| " ".to_string());
                    let mut expr = rcvr;
                    while let Node::Expr(call_expr) = self.tcx.parent_hir_node(expr.hir_id)
                        && let hir::ExprKind::MethodCall(hir::PathSegment { .. }, ..) =
                            call_expr.kind
                    {
                        expr = call_expr;
                    }
                    match self.tcx.parent_hir_node(expr.hir_id) {
                        Node::LetStmt(stmt)
                            if let Some(init) = stmt.init
                                && let Ok(code) =
                                    self.tcx.sess.source_map().span_to_snippet(rcvr.span) =>
                        {
                            // We need to take care to account for the existing binding when we
                            // suggest the code.
                            err.multipart_suggestion(
                                "consider pinning the expression",
                                vec![
                                    (
                                        stmt.span.shrink_to_lo(),
                                        format!(
                                            "let mut pinned = std::pin::pin!({code});\n{indent}"
                                        ),
                                    ),
                                    (
                                        init.span.until(rcvr.span.shrink_to_hi()),
                                        format!("pinned.{pin_call}()"),
                                    ),
                                ],
                                Applicability::MaybeIncorrect,
                            );
                        }
                        Node::Block(_) | Node::Stmt(_) => {
                            // There's no binding, so we can provide a slightly nicer looking
                            // suggestion.
                            err.multipart_suggestion(
                                "consider pinning the expression",
                                vec![
                                    (
                                        rcvr.span.shrink_to_lo(),
                                        format!("let mut pinned = std::pin::pin!("),
                                    ),
                                    (
                                        rcvr.span.shrink_to_hi(),
                                        format!(");\n{indent}pinned.{pin_call}()"),
                                    ),
                                ],
                                Applicability::MaybeIncorrect,
                            );
                        }
                        _ => {
                            // We don't quite know what the users' code looks like, so we don't
                            // provide a pinning suggestion.
                            err.span_help(
                                rcvr.span,
                                "consider pinning the expression with `std::pin::pin!()` and \
                                 assigning that to a new binding",
                            );
                        }
                    }
                    // We don't care about the other suggestions.
                    alt_rcvr_sugg = true;
                }
            }
        }

        if let SelfSource::QPath(ty) = source
            && !valid_out_of_scope_traits.is_empty()
            && let hir::TyKind::Path(path) = ty.kind
            && let hir::QPath::Resolved(..) = path
            && let Some(assoc) = self
                .tcx
                .associated_items(valid_out_of_scope_traits[0])
                .filter_by_name_unhygienic(item_name.name)
                .next()
        {
            // See if the `Type::function(val)` where `function` wasn't found corresponds to a
            // `Trait` that is imported directly, but `Type` came from a different version of the
            // same crate.

            let rcvr_ty = self.node_ty_opt(ty.hir_id);
            trait_in_other_version_found = self
                .detect_and_explain_multiple_crate_versions_of_trait_item(
                    err,
                    assoc.def_id,
                    ty.hir_id,
                    rcvr_ty,
                );
        }
        if !trait_in_other_version_found
            && self.suggest_valid_traits(err, item_name, valid_out_of_scope_traits, true)
        {
            return;
        }

        let type_is_local = self.type_derefs_to_local(span, rcvr_ty, source);

        let mut arbitrary_rcvr = vec![];
        // There are no traits implemented, so lets suggest some traits to
        // implement, by finding ones that have the item name, and are
        // legal to implement.
        let mut candidates = all_traits(self.tcx)
            .into_iter()
            // Don't issue suggestions for unstable traits since they're
            // unlikely to be implementable anyway
            .filter(|info| match self.tcx.lookup_stability(info.def_id) {
                Some(attr) => attr.level.is_stable(),
                None => true,
            })
            .filter(|info| {
                // Static candidates are already implemented, and known not to work
                // Do not suggest them again
                static_candidates.iter().all(|sc| match *sc {
                    CandidateSource::Trait(def_id) => def_id != info.def_id,
                    CandidateSource::Impl(def_id) => {
                        self.tcx.trait_id_of_impl(def_id) != Some(info.def_id)
                    }
                })
            })
            .filter(|info| {
                // We approximate the coherence rules to only suggest
                // traits that are legal to implement by requiring that
                // either the type or trait is local. Multi-dispatch means
                // this isn't perfect (that is, there are cases when
                // implementing a trait would be legal but is rejected
                // here).
                (type_is_local || info.def_id.is_local())
                    && !self.tcx.trait_is_auto(info.def_id)
                    && self
                        .associated_value(info.def_id, item_name)
                        .filter(|item| {
                            if item.is_fn() {
                                let id = item
                                    .def_id
                                    .as_local()
                                    .map(|def_id| self.tcx.hir_node_by_def_id(def_id));
                                if let Some(hir::Node::TraitItem(hir::TraitItem {
                                    kind: hir::TraitItemKind::Fn(fn_sig, method),
                                    ..
                                })) = id
                                {
                                    let self_first_arg = match method {
                                        hir::TraitFn::Required([ident, ..]) => {
                                            matches!(ident, Some(Ident { name: kw::SelfLower, .. }))
                                        }
                                        hir::TraitFn::Provided(body_id) => {
                                            self.tcx.hir_body(*body_id).params.first().is_some_and(
                                                |param| {
                                                    matches!(
                                                        param.pat.kind,
                                                        hir::PatKind::Binding(_, _, ident, _)
                                                            if ident.name == kw::SelfLower
                                                    )
                                                },
                                            )
                                        }
                                        _ => false,
                                    };

                                    if !fn_sig.decl.implicit_self.has_implicit_self()
                                        && self_first_arg
                                    {
                                        if let Some(ty) = fn_sig.decl.inputs.get(0) {
                                            arbitrary_rcvr.push(ty.span);
                                        }
                                        return false;
                                    }
                                }
                            }
                            // We only want to suggest public or local traits (#45781).
                            item.visibility(self.tcx).is_public() || info.def_id.is_local()
                        })
                        .is_some()
            })
            .collect::<Vec<_>>();
        for span in &arbitrary_rcvr {
            err.span_label(
                *span,
                "the method might not be found because of this arbitrary self type",
            );
        }
        if alt_rcvr_sugg {
            return;
        }

        if !candidates.is_empty() {
            // Sort local crate results before others
            candidates
                .sort_by_key(|&info| (!info.def_id.is_local(), self.tcx.def_path_str(info.def_id)));
            candidates.dedup();

            let param_type = match *rcvr_ty.kind() {
                ty::Param(param) => Some(param),
                ty::Ref(_, ty, _) => match *ty.kind() {
                    ty::Param(param) => Some(param),
                    _ => None,
                },
                _ => None,
            };
            if !trait_missing_method {
                err.help(if param_type.is_some() {
                    "items from traits can only be used if the type parameter is bounded by the trait"
                } else {
                    "items from traits can only be used if the trait is implemented and in scope"
                });
            }

            let candidates_len = candidates.len();
            let message = |action| {
                format!(
                    "the following {traits_define} an item `{name}`, perhaps you need to {action} \
                     {one_of_them}:",
                    traits_define =
                        if candidates_len == 1 { "trait defines" } else { "traits define" },
                    action = action,
                    one_of_them = if candidates_len == 1 { "it" } else { "one of them" },
                    name = item_name,
                )
            };
            // Obtain the span for `param` and use it for a structured suggestion.
            if let Some(param) = param_type {
                let generics = self.tcx.generics_of(self.body_id.to_def_id());
                let type_param = generics.type_param(param, self.tcx);
                let tcx = self.tcx;
                if let Some(def_id) = type_param.def_id.as_local() {
                    let id = tcx.local_def_id_to_hir_id(def_id);
                    // Get the `hir::Param` to verify whether it already has any bounds.
                    // We do this to avoid suggesting code that ends up as `T: FooBar`,
                    // instead we suggest `T: Foo + Bar` in that case.
                    match tcx.hir_node(id) {
                        Node::GenericParam(param) => {
                            enum Introducer {
                                Plus,
                                Colon,
                                Nothing,
                            }
                            let hir_generics = tcx.hir_get_generics(id.owner.def_id).unwrap();
                            let trait_def_ids: DefIdSet = hir_generics
                                .bounds_for_param(def_id)
                                .flat_map(|bp| bp.bounds.iter())
                                .filter_map(|bound| bound.trait_ref()?.trait_def_id())
                                .collect();
                            if candidates.iter().any(|t| trait_def_ids.contains(&t.def_id)) {
                                return;
                            }
                            let msg = message(format!(
                                "restrict type parameter `{}` with",
                                param.name.ident(),
                            ));
                            let bounds_span = hir_generics.bounds_span_for_suggestions(def_id);
                            let mut applicability = Applicability::MaybeIncorrect;
                            // Format the path of each suggested candidate, providing placeholders
                            // for any generic arguments without defaults.
                            let candidate_strs: Vec<_> = candidates
                                .iter()
                                .map(|cand| {
                                    let cand_path = tcx.def_path_str(cand.def_id);
                                    let cand_params = &tcx.generics_of(cand.def_id).own_params;
                                    let cand_args: String = cand_params
                                        .iter()
                                        .skip(1)
                                        .filter_map(|param| match param.kind {
                                            ty::GenericParamDefKind::Type {
                                                has_default: true,
                                                ..
                                            }
                                            | ty::GenericParamDefKind::Const {
                                                has_default: true,
                                                ..
                                            } => None,
                                            _ => Some(param.name.as_str()),
                                        })
                                        .intersperse(", ")
                                        .collect();
                                    if cand_args.is_empty() {
                                        cand_path
                                    } else {
                                        applicability = Applicability::HasPlaceholders;
                                        format!("{cand_path}</* {cand_args} */>")
                                    }
                                })
                                .collect();

                            if rcvr_ty.is_ref()
                                && param.is_impl_trait()
                                && let Some((bounds_span, _)) = bounds_span
                            {
                                err.multipart_suggestions(
                                    msg,
                                    candidate_strs.iter().map(|cand| {
                                        vec![
                                            (param.span.shrink_to_lo(), "(".to_string()),
                                            (bounds_span, format!(" + {cand})")),
                                        ]
                                    }),
                                    applicability,
                                );
                                return;
                            }

                            let (sp, introducer, open_paren_sp) =
                                if let Some((span, open_paren_sp)) = bounds_span {
                                    (span, Introducer::Plus, open_paren_sp)
                                } else if let Some(colon_span) = param.colon_span {
                                    (colon_span.shrink_to_hi(), Introducer::Nothing, None)
                                } else if param.is_impl_trait() {
                                    (param.span.shrink_to_hi(), Introducer::Plus, None)
                                } else {
                                    (param.span.shrink_to_hi(), Introducer::Colon, None)
                                };

                            let all_suggs = candidate_strs.iter().map(|cand| {
                                let suggestion = format!(
                                    "{} {cand}",
                                    match introducer {
                                        Introducer::Plus => " +",
                                        Introducer::Colon => ":",
                                        Introducer::Nothing => "",
                                    },
                                );

                                let mut suggs = vec![];

                                if let Some(open_paren_sp) = open_paren_sp {
                                    suggs.push((open_paren_sp, "(".to_string()));
                                    suggs.push((sp, format!("){suggestion}")));
                                } else {
                                    suggs.push((sp, suggestion));
                                }

                                suggs
                            });

                            err.multipart_suggestions(msg, all_suggs, applicability);

                            return;
                        }
                        Node::Item(hir::Item {
                            kind: hir::ItemKind::Trait(_, _, ident, _, bounds, _),
                            ..
                        }) => {
                            let (sp, sep, article) = if bounds.is_empty() {
                                (ident.span.shrink_to_hi(), ":", "a")
                            } else {
                                (bounds.last().unwrap().span().shrink_to_hi(), " +", "another")
                            };
                            err.span_suggestions(
                                sp,
                                message(format!("add {article} supertrait for")),
                                candidates
                                    .iter()
                                    .map(|t| format!("{} {}", sep, tcx.def_path_str(t.def_id),)),
                                Applicability::MaybeIncorrect,
                            );
                            return;
                        }
                        _ => {}
                    }
                }
            }

            let (potential_candidates, explicitly_negative) = if param_type.is_some() {
                // FIXME: Even though negative bounds are not implemented, we could maybe handle
                // cases where a positive bound implies a negative impl.
                (candidates, Vec::new())
            } else if let Some(simp_rcvr_ty) =
                simplify_type(self.tcx, rcvr_ty, TreatParams::AsRigid)
            {
                let mut potential_candidates = Vec::new();
                let mut explicitly_negative = Vec::new();
                for candidate in candidates {
                    // Check if there's a negative impl of `candidate` for `rcvr_ty`
                    if self
                        .tcx
                        .all_impls(candidate.def_id)
                        .map(|imp_did| {
                            self.tcx.impl_trait_header(imp_did).expect(
                                "inherent impls can't be candidates, only trait impls can be",
                            )
                        })
                        .filter(|header| header.polarity != ty::ImplPolarity::Positive)
                        .any(|header| {
                            let imp = header.trait_ref.instantiate_identity();
                            let imp_simp =
                                simplify_type(self.tcx, imp.self_ty(), TreatParams::AsRigid);
                            imp_simp.is_some_and(|s| s == simp_rcvr_ty)
                        })
                    {
                        explicitly_negative.push(candidate);
                    } else {
                        potential_candidates.push(candidate);
                    }
                }
                (potential_candidates, explicitly_negative)
            } else {
                // We don't know enough about `recv_ty` to make proper suggestions.
                (candidates, Vec::new())
            };

            let impls_trait = |def_id: DefId| {
                let args = ty::GenericArgs::for_item(self.tcx, def_id, |param, _| {
                    if param.index == 0 {
                        rcvr_ty.into()
                    } else {
                        self.infcx.var_for_def(span, param)
                    }
                });
                self.infcx
                    .type_implements_trait(def_id, args, self.param_env)
                    .must_apply_modulo_regions()
                    && param_type.is_none()
            };
            match &potential_candidates[..] {
                [] => {}
                [trait_info] if trait_info.def_id.is_local() => {
                    if impls_trait(trait_info.def_id) {
                        self.suggest_valid_traits(err, item_name, vec![trait_info.def_id], false);
                    } else {
                        err.subdiagnostic(CandidateTraitNote {
                            span: self.tcx.def_span(trait_info.def_id),
                            trait_name: self.tcx.def_path_str(trait_info.def_id),
                            item_name,
                            action_or_ty: if trait_missing_method {
                                "NONE".to_string()
                            } else {
                                param_type.map_or_else(
                                    || "implement".to_string(), // FIXME: it might only need to be imported into scope, not implemented.
                                    |p| p.to_string(),
                                )
                            },
                        });
                    }
                }
                trait_infos => {
                    let mut msg = message(param_type.map_or_else(
                        || "implement".to_string(), // FIXME: it might only need to be imported into scope, not implemented.
                        |param| format!("restrict type parameter `{param}` with"),
                    ));
                    for (i, trait_info) in trait_infos.iter().enumerate() {
                        if impls_trait(trait_info.def_id) {
                            self.suggest_valid_traits(
                                err,
                                item_name,
                                vec![trait_info.def_id],
                                false,
                            );
                        }
                        msg.push_str(&format!(
                            "\ncandidate #{}: `{}`",
                            i + 1,
                            self.tcx.def_path_str(trait_info.def_id),
                        ));
                    }
                    err.note(msg);
                }
            }
            match &explicitly_negative[..] {
                [] => {}
                [trait_info] => {
                    let msg = format!(
                        "the trait `{}` defines an item `{}`, but is explicitly unimplemented",
                        self.tcx.def_path_str(trait_info.def_id),
                        item_name
                    );
                    err.note(msg);
                }
                trait_infos => {
                    let mut msg = format!(
                        "the following traits define an item `{item_name}`, but are explicitly unimplemented:"
                    );
                    for trait_info in trait_infos {
                        msg.push_str(&format!("\n{}", self.tcx.def_path_str(trait_info.def_id)));
                    }
                    err.note(msg);
                }
            }
        }
    }

    fn detect_and_explain_multiple_crate_versions_of_trait_item(
        &self,
        err: &mut Diag<'_>,
        item_def_id: DefId,
        hir_id: hir::HirId,
        rcvr_ty: Option<Ty<'_>>,
    ) -> bool {
        let hir_id = self.tcx.parent_hir_id(hir_id);
        let Some(traits) = self.tcx.in_scope_traits(hir_id) else { return false };
        if traits.is_empty() {
            return false;
        }
        let trait_def_id = self.tcx.parent(item_def_id);
        if !self.tcx.is_trait(trait_def_id) {
            return false;
        }
        let krate = self.tcx.crate_name(trait_def_id.krate);
        let name = self.tcx.item_name(trait_def_id);
        let candidates: Vec<_> = traits
            .iter()
            .filter(|c| {
                c.def_id.krate != trait_def_id.krate
                    && self.tcx.crate_name(c.def_id.krate) == krate
                    && self.tcx.item_name(c.def_id) == name
            })
            .map(|c| (c.def_id, c.import_ids.get(0).cloned()))
            .collect();
        if candidates.is_empty() {
            return false;
        }
        let item_span = self.tcx.def_span(item_def_id);
        let msg = format!(
            "there are multiple different versions of crate `{krate}` in the dependency graph",
        );
        let trait_span = self.tcx.def_span(trait_def_id);
        let mut multi_span: MultiSpan = trait_span.into();
        multi_span.push_span_label(trait_span, format!("this is the trait that is needed"));
        let descr = self.tcx.associated_item(item_def_id).descr();
        let rcvr_ty =
            rcvr_ty.map(|t| format!("`{t}`")).unwrap_or_else(|| "the receiver".to_string());
        multi_span
            .push_span_label(item_span, format!("the {descr} is available for {rcvr_ty} here"));
        for (def_id, import_def_id) in candidates {
            if let Some(import_def_id) = import_def_id {
                multi_span.push_span_label(
                    self.tcx.def_span(import_def_id),
                    format!(
                        "`{name}` imported here doesn't correspond to the right version of crate \
                         `{krate}`",
                    ),
                );
            }
            multi_span.push_span_label(
                self.tcx.def_span(def_id),
                format!("this is the trait that was imported"),
            );
        }
        err.span_note(multi_span, msg);
        true
    }

    /// issue #102320, for `unwrap_or` with closure as argument, suggest `unwrap_or_else`
    /// FIXME: currently not working for suggesting `map_or_else`, see #102408
    pub(crate) fn suggest_else_fn_with_closure(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        found: Ty<'tcx>,
        expected: Ty<'tcx>,
    ) -> bool {
        let Some((_def_id_or_name, output, _inputs)) = self.extract_callable_info(found) else {
            return false;
        };

        if !self.may_coerce(output, expected) {
            return false;
        }

        if let Node::Expr(call_expr) = self.tcx.parent_hir_node(expr.hir_id)
            && let hir::ExprKind::MethodCall(
                hir::PathSegment { ident: method_name, .. },
                self_expr,
                args,
                ..,
            ) = call_expr.kind
            && let Some(self_ty) = self.typeck_results.borrow().expr_ty_opt(self_expr)
        {
            let new_name = Ident {
                name: Symbol::intern(&format!("{}_else", method_name.as_str())),
                span: method_name.span,
            };
            let probe = self.lookup_probe_for_diagnostic(
                new_name,
                self_ty,
                self_expr,
                ProbeScope::TraitsInScope,
                Some(expected),
            );

            // check the method arguments number
            if let Ok(pick) = probe
                && let fn_sig = self.tcx.fn_sig(pick.item.def_id)
                && let fn_args = fn_sig.skip_binder().skip_binder().inputs()
                && fn_args.len() == args.len() + 1
            {
                err.span_suggestion_verbose(
                    method_name.span.shrink_to_hi(),
                    format!("try calling `{}` instead", new_name.name.as_str()),
                    "_else",
                    Applicability::MaybeIncorrect,
                );
                return true;
            }
        }
        false
    }

    /// Checks whether there is a local type somewhere in the chain of
    /// autoderefs of `rcvr_ty`.
    fn type_derefs_to_local(
        &self,
        span: Span,
        rcvr_ty: Ty<'tcx>,
        source: SelfSource<'tcx>,
    ) -> bool {
        fn is_local(ty: Ty<'_>) -> bool {
            match ty.kind() {
                ty::Adt(def, _) => def.did().is_local(),
                ty::Foreign(did) => did.is_local(),
                ty::Dynamic(tr, ..) => tr.principal().is_some_and(|d| d.def_id().is_local()),
                ty::Param(_) => true,

                // Everything else (primitive types, etc.) is effectively
                // non-local (there are "edge" cases, e.g., `(LocalType,)`, but
                // the noise from these sort of types is usually just really
                // annoying, rather than any sort of help).
                _ => false,
            }
        }

        // This occurs for UFCS desugaring of `T::method`, where there is no
        // receiver expression for the method call, and thus no autoderef.
        if let SelfSource::QPath(_) = source {
            return is_local(rcvr_ty);
        }

        self.autoderef(span, rcvr_ty).silence_errors().any(|(ty, _)| is_local(ty))
    }
}

#[derive(Copy, Clone, Debug)]
enum SelfSource<'a> {
    QPath(&'a hir::Ty<'a>),
    MethodCall(&'a hir::Expr<'a> /* rcvr */),
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct TraitInfo {
    pub def_id: DefId,
}

/// Retrieves all traits in this crate and any dependent crates,
/// and wraps them into `TraitInfo` for custom sorting.
pub(crate) fn all_traits(tcx: TyCtxt<'_>) -> Vec<TraitInfo> {
    tcx.all_traits().map(|def_id| TraitInfo { def_id }).collect()
}

fn print_disambiguation_help<'tcx>(
    tcx: TyCtxt<'tcx>,
    err: &mut Diag<'_>,
    source: SelfSource<'tcx>,
    args: Option<&'tcx [hir::Expr<'tcx>]>,
    trait_ref: ty::TraitRef<'tcx>,
    candidate_idx: Option<usize>,
    span: Span,
    item: ty::AssocItem,
) -> Option<String> {
    let trait_impl_type = trait_ref.self_ty().peel_refs();
    let trait_ref = if item.is_method() {
        trait_ref.print_only_trait_name().to_string()
    } else {
        format!("<{} as {}>", trait_ref.args[0], trait_ref.print_only_trait_name())
    };
    Some(
        if item.is_fn()
            && let SelfSource::MethodCall(receiver) = source
            && let Some(args) = args
        {
            let def_kind_descr = tcx.def_kind_descr(item.as_def_kind(), item.def_id);
            let item_name = item.ident(tcx);
            let first_input =
                tcx.fn_sig(item.def_id).instantiate_identity().skip_binder().inputs().get(0);
            let (first_arg_type, rcvr_ref) = (
                first_input.map(|first| first.peel_refs()),
                first_input
                    .and_then(|ty| ty.ref_mutability())
                    .map_or("", |mutbl| mutbl.ref_prefix_str()),
            );

            // If the type of first arg of this assoc function is `Self` or current trait impl type or `arbitrary_self_types`, we need to take the receiver as args. Otherwise, we don't.
            let args = if let Some(first_arg_type) = first_arg_type
                && (first_arg_type == tcx.types.self_param
                    || first_arg_type == trait_impl_type
                    || item.is_method())
            {
                Some(receiver)
            } else {
                None
            }
            .into_iter()
            .chain(args)
            .map(|arg| {
                tcx.sess.source_map().span_to_snippet(arg.span).unwrap_or_else(|_| "_".to_owned())
            })
            .collect::<Vec<_>>()
            .join(", ");

            let args = format!("({}{})", rcvr_ref, args);
            err.span_suggestion_verbose(
                span,
                format!(
                    "disambiguate the {def_kind_descr} for {}",
                    if let Some(candidate) = candidate_idx {
                        format!("candidate #{candidate}")
                    } else {
                        "the candidate".to_string()
                    },
                ),
                format!("{trait_ref}::{item_name}{args}"),
                Applicability::HasPlaceholders,
            );
            return None;
        } else {
            format!("{trait_ref}::")
        },
    )
}
