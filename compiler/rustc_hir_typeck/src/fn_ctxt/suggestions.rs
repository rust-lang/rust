use core::cmp::min;
use core::iter;

use hir::def_id::LocalDefId;
use rustc_ast::util::parser::ExprPrecedence;
use rustc_data_structures::packed::Pu128;
use rustc_errors::{Applicability, Diag, MultiSpan, listify};
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Res};
use rustc_hir::lang_items::LangItem;
use rustc_hir::{
    self as hir, Arm, CoroutineDesugaring, CoroutineKind, CoroutineSource, Expr, ExprKind,
    GenericBound, HirId, Node, PatExpr, PatExprKind, Path, QPath, Stmt, StmtKind, TyKind,
    WherePredicateKind, expr_needs_parens,
};
use rustc_hir_analysis::hir_ty_lowering::HirTyLowerer;
use rustc_hir_analysis::suggest_impl_trait;
use rustc_middle::middle::stability::EvalResult;
use rustc_middle::span_bug;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{
    self, Article, Binder, IsSuggestable, Ty, TyCtxt, TypeVisitableExt, Upcast,
    suggest_constraining_type_params,
};
use rustc_session::errors::ExprParenthesesNeeded;
use rustc_span::source_map::Spanned;
use rustc_span::{ExpnKind, Ident, MacroKind, Span, Symbol, sym};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::error_reporting::traits::DefIdOrName;
use rustc_trait_selection::infer::InferCtxtExt;
use rustc_trait_selection::traits;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt as _;
use tracing::{debug, instrument};

use super::FnCtxt;
use crate::fn_ctxt::rustc_span::BytePos;
use crate::method::probe;
use crate::method::probe::{IsSuggestion, Mode, ProbeScope};
use crate::{errors, fluent_generated as fluent};

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub(crate) fn body_fn_sig(&self) -> Option<ty::FnSig<'tcx>> {
        self.typeck_results
            .borrow()
            .liberated_fn_sigs()
            .get(self.tcx.local_def_id_to_hir_id(self.body_id))
            .copied()
    }

    pub(in super::super) fn suggest_semicolon_at_end(&self, span: Span, err: &mut Diag<'_>) {
        // This suggestion is incorrect for
        // fn foo() -> bool { match () { () => true } || match () { () => true } }
        err.span_suggestion_short(
            span.shrink_to_hi(),
            "consider using a semicolon here",
            ";",
            Applicability::MaybeIncorrect,
        );
    }

    /// On implicit return expressions with mismatched types, provides the following suggestions:
    ///
    /// - Points out the method's return type as the reason for the expected type.
    /// - Possible missing semicolon.
    /// - Possible missing return type if the return type is the default, and not `fn main()`.
    pub(crate) fn suggest_mismatched_types_on_tail(
        &self,
        err: &mut Diag<'_>,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        blk_id: HirId,
    ) -> bool {
        let expr = expr.peel_drop_temps();
        let mut pointing_at_return_type = false;
        if let hir::ExprKind::Break(..) = expr.kind {
            // `break` type mismatches provide better context for tail `loop` expressions.
            return false;
        }
        if let Some((fn_id, fn_decl)) = self.get_fn_decl(blk_id) {
            pointing_at_return_type =
                self.suggest_missing_return_type(err, fn_decl, expected, found, fn_id);
            self.suggest_missing_break_or_return_expr(
                err, expr, fn_decl, expected, found, blk_id, fn_id,
            );
        }
        pointing_at_return_type
    }

    /// When encountering an fn-like type, try accessing the output of the type
    /// and suggesting calling it if it satisfies a predicate (i.e. if the
    /// output has a method or a field):
    /// ```compile_fail,E0308
    /// fn foo(x: usize) -> usize { x }
    /// let x: usize = foo;  // suggest calling the `foo` function: `foo(42)`
    /// ```
    pub(crate) fn suggest_fn_call(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        found: Ty<'tcx>,
        can_satisfy: impl FnOnce(Ty<'tcx>) -> bool,
    ) -> bool {
        let Some((def_id_or_name, output, inputs)) = self.extract_callable_info(found) else {
            return false;
        };
        if can_satisfy(output) {
            let (sugg_call, mut applicability) = match inputs.len() {
                0 => ("".to_string(), Applicability::MachineApplicable),
                1..=4 => (
                    inputs
                        .iter()
                        .map(|ty| {
                            if ty.is_suggestable(self.tcx, false) {
                                format!("/* {ty} */")
                            } else {
                                "/* value */".to_string()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", "),
                    Applicability::HasPlaceholders,
                ),
                _ => ("/* ... */".to_string(), Applicability::HasPlaceholders),
            };

            let msg = match def_id_or_name {
                DefIdOrName::DefId(def_id) => match self.tcx.def_kind(def_id) {
                    DefKind::Ctor(CtorOf::Struct, _) => "construct this tuple struct".to_string(),
                    DefKind::Ctor(CtorOf::Variant, _) => "construct this tuple variant".to_string(),
                    kind => format!("call this {}", self.tcx.def_kind_descr(kind, def_id)),
                },
                DefIdOrName::Name(name) => format!("call this {name}"),
            };

            let sugg = match expr.kind {
                hir::ExprKind::Call(..)
                | hir::ExprKind::Path(..)
                | hir::ExprKind::Index(..)
                | hir::ExprKind::Lit(..) => {
                    vec![(expr.span.shrink_to_hi(), format!("({sugg_call})"))]
                }
                hir::ExprKind::Closure { .. } => {
                    // Might be `{ expr } || { bool }`
                    applicability = Applicability::MaybeIncorrect;
                    vec![
                        (expr.span.shrink_to_lo(), "(".to_string()),
                        (expr.span.shrink_to_hi(), format!(")({sugg_call})")),
                    ]
                }
                _ => {
                    vec![
                        (expr.span.shrink_to_lo(), "(".to_string()),
                        (expr.span.shrink_to_hi(), format!(")({sugg_call})")),
                    ]
                }
            };

            err.multipart_suggestion_verbose(
                format!("use parentheses to {msg}"),
                sugg,
                applicability,
            );
            return true;
        }
        false
    }

    /// Extracts information about a callable type for diagnostics. This is a
    /// heuristic -- it doesn't necessarily mean that a type is always callable,
    /// because the callable type must also be well-formed to be called.
    pub(in super::super) fn extract_callable_info(
        &self,
        ty: Ty<'tcx>,
    ) -> Option<(DefIdOrName, Ty<'tcx>, Vec<Ty<'tcx>>)> {
        self.err_ctxt().extract_callable_info(self.body_id, self.param_env, ty)
    }

    pub(crate) fn suggest_two_fn_call(
        &self,
        err: &mut Diag<'_>,
        lhs_expr: &'tcx hir::Expr<'tcx>,
        lhs_ty: Ty<'tcx>,
        rhs_expr: &'tcx hir::Expr<'tcx>,
        rhs_ty: Ty<'tcx>,
        can_satisfy: impl FnOnce(Ty<'tcx>, Ty<'tcx>) -> bool,
    ) -> bool {
        if lhs_expr.span.in_derive_expansion() || rhs_expr.span.in_derive_expansion() {
            return false;
        }
        let Some((_, lhs_output_ty, lhs_inputs)) = self.extract_callable_info(lhs_ty) else {
            return false;
        };
        let Some((_, rhs_output_ty, rhs_inputs)) = self.extract_callable_info(rhs_ty) else {
            return false;
        };

        if can_satisfy(lhs_output_ty, rhs_output_ty) {
            let mut sugg = vec![];
            let mut applicability = Applicability::MachineApplicable;

            for (expr, inputs) in [(lhs_expr, lhs_inputs), (rhs_expr, rhs_inputs)] {
                let (sugg_call, this_applicability) = match inputs.len() {
                    0 => ("".to_string(), Applicability::MachineApplicable),
                    1..=4 => (
                        inputs
                            .iter()
                            .map(|ty| {
                                if ty.is_suggestable(self.tcx, false) {
                                    format!("/* {ty} */")
                                } else {
                                    "/* value */".to_string()
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(", "),
                        Applicability::HasPlaceholders,
                    ),
                    _ => ("/* ... */".to_string(), Applicability::HasPlaceholders),
                };

                applicability = applicability.max(this_applicability);

                match expr.kind {
                    hir::ExprKind::Call(..)
                    | hir::ExprKind::Path(..)
                    | hir::ExprKind::Index(..)
                    | hir::ExprKind::Lit(..) => {
                        sugg.extend([(expr.span.shrink_to_hi(), format!("({sugg_call})"))]);
                    }
                    hir::ExprKind::Closure { .. } => {
                        // Might be `{ expr } || { bool }`
                        applicability = Applicability::MaybeIncorrect;
                        sugg.extend([
                            (expr.span.shrink_to_lo(), "(".to_string()),
                            (expr.span.shrink_to_hi(), format!(")({sugg_call})")),
                        ]);
                    }
                    _ => {
                        sugg.extend([
                            (expr.span.shrink_to_lo(), "(".to_string()),
                            (expr.span.shrink_to_hi(), format!(")({sugg_call})")),
                        ]);
                    }
                }
            }

            err.multipart_suggestion_verbose("use parentheses to call these", sugg, applicability);

            true
        } else {
            false
        }
    }

    pub(crate) fn suggest_remove_last_method_call(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'tcx>,
        expected: Ty<'tcx>,
    ) -> bool {
        if let hir::ExprKind::MethodCall(hir::PathSegment { ident: method, .. }, recv_expr, &[], _) =
            expr.kind
            && let Some(recv_ty) = self.typeck_results.borrow().expr_ty_opt(recv_expr)
            && self.may_coerce(recv_ty, expected)
            && let name = method.name.as_str()
            && (name.starts_with("to_") || name.starts_with("as_") || name == "into")
        {
            let span = if let Some(recv_span) = recv_expr.span.find_ancestor_inside(expr.span) {
                expr.span.with_lo(recv_span.hi())
            } else {
                expr.span.with_lo(method.span.lo() - rustc_span::BytePos(1))
            };
            err.span_suggestion_verbose(
                span,
                "try removing the method call",
                "",
                Applicability::MachineApplicable,
            );
            return true;
        }
        false
    }

    pub(crate) fn suggest_deref_ref_or_into(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'tcx>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        expected_ty_expr: Option<&'tcx hir::Expr<'tcx>>,
    ) -> bool {
        let expr = expr.peel_blocks();
        let methods =
            self.get_conversion_methods_for_diagnostic(expr.span, expected, found, expr.hir_id);

        if let Some((suggestion, msg, applicability, verbose, annotation)) =
            self.suggest_deref_or_ref(expr, found, expected)
        {
            if verbose {
                err.multipart_suggestion_verbose(msg, suggestion, applicability);
            } else {
                err.multipart_suggestion(msg, suggestion, applicability);
            }
            if annotation {
                let suggest_annotation = match expr.peel_drop_temps().kind {
                    hir::ExprKind::AddrOf(hir::BorrowKind::Ref, mutbl, _) => mutbl.ref_prefix_str(),
                    _ => return true,
                };
                let mut tuple_indexes = Vec::new();
                let mut expr_id = expr.hir_id;
                for (parent_id, node) in self.tcx.hir_parent_iter(expr.hir_id) {
                    match node {
                        Node::Expr(&Expr { kind: ExprKind::Tup(subs), .. }) => {
                            tuple_indexes.push(
                                subs.iter()
                                    .enumerate()
                                    .find(|(_, sub_expr)| sub_expr.hir_id == expr_id)
                                    .unwrap()
                                    .0,
                            );
                            expr_id = parent_id;
                        }
                        Node::LetStmt(local) => {
                            if let Some(mut ty) = local.ty {
                                while let Some(index) = tuple_indexes.pop() {
                                    match ty.kind {
                                        TyKind::Tup(tys) => ty = &tys[index],
                                        _ => return true,
                                    }
                                }
                                let annotation_span = ty.span;
                                err.span_suggestion(
                                    annotation_span.with_hi(annotation_span.lo()),
                                    "alternatively, consider changing the type annotation",
                                    suggest_annotation,
                                    Applicability::MaybeIncorrect,
                                );
                            }
                            break;
                        }
                        _ => break,
                    }
                }
            }
            return true;
        }

        if self.suggest_else_fn_with_closure(err, expr, found, expected) {
            return true;
        }

        if self.suggest_fn_call(err, expr, found, |output| self.may_coerce(output, expected))
            && let ty::FnDef(def_id, ..) = *found.kind()
            && let Some(sp) = self.tcx.hir_span_if_local(def_id)
        {
            let name = self.tcx.item_name(def_id);
            let kind = self.tcx.def_kind(def_id);
            if let DefKind::Ctor(of, CtorKind::Fn) = kind {
                err.span_label(
                    sp,
                    format!(
                        "`{name}` defines {} constructor here, which should be called",
                        match of {
                            CtorOf::Struct => "a struct",
                            CtorOf::Variant => "an enum variant",
                        }
                    ),
                );
            } else {
                let descr = self.tcx.def_kind_descr(kind, def_id);
                err.span_label(sp, format!("{descr} `{name}` defined here"));
            }
            return true;
        }

        if self.suggest_cast(err, expr, found, expected, expected_ty_expr) {
            return true;
        }

        if !methods.is_empty() {
            let mut suggestions = methods
                .iter()
                .filter_map(|conversion_method| {
                    let conversion_method_name = conversion_method.name();
                    let receiver_method_ident = expr.method_ident();
                    if let Some(method_ident) = receiver_method_ident
                        && method_ident.name == conversion_method_name
                    {
                        return None; // do not suggest code that is already there (#53348)
                    }

                    let method_call_list = [sym::to_vec, sym::to_string];
                    let mut sugg = if let ExprKind::MethodCall(receiver_method, ..) = expr.kind
                        && receiver_method.ident.name == sym::clone
                        && method_call_list.contains(&conversion_method_name)
                    // If receiver is `.clone()` and found type has one of those methods,
                    // we guess that the user wants to convert from a slice type (`&[]` or `&str`)
                    // to an owned type (`Vec` or `String`). These conversions clone internally,
                    // so we remove the user's `clone` call.
                    {
                        vec![(receiver_method.ident.span, conversion_method_name.to_string())]
                    } else if expr.precedence() < ExprPrecedence::Unambiguous {
                        vec![
                            (expr.span.shrink_to_lo(), "(".to_string()),
                            (expr.span.shrink_to_hi(), format!(").{}()", conversion_method_name)),
                        ]
                    } else {
                        vec![(expr.span.shrink_to_hi(), format!(".{}()", conversion_method_name))]
                    };
                    let struct_pat_shorthand_field =
                        self.tcx.hir_maybe_get_struct_pattern_shorthand_field(expr);
                    if let Some(name) = struct_pat_shorthand_field {
                        sugg.insert(0, (expr.span.shrink_to_lo(), format!("{name}: ")));
                    }
                    Some(sugg)
                })
                .peekable();
            if suggestions.peek().is_some() {
                err.multipart_suggestions(
                    "try using a conversion method",
                    suggestions,
                    Applicability::MaybeIncorrect,
                );
                return true;
            }
        }

        if let Some((found_ty_inner, expected_ty_inner, error_tys)) =
            self.deconstruct_option_or_result(found, expected)
            && let ty::Ref(_, peeled, hir::Mutability::Not) = *expected_ty_inner.kind()
        {
            // Suggest removing any stray borrows (unless there's macro shenanigans involved).
            let inner_expr = expr.peel_borrows();
            if !inner_expr.span.eq_ctxt(expr.span) {
                return false;
            }
            let borrow_removal_span = if inner_expr.hir_id == expr.hir_id {
                None
            } else {
                Some(expr.span.shrink_to_lo().until(inner_expr.span))
            };
            // Given `Result<_, E>`, check our expected ty is `Result<_, &E>` for
            // `as_ref` and `as_deref` compatibility.
            let error_tys_equate_as_ref = error_tys.is_none_or(|(found, expected)| {
                self.can_eq(
                    self.param_env,
                    Ty::new_imm_ref(self.tcx, self.tcx.lifetimes.re_erased, found),
                    expected,
                )
            });

            let prefix_wrap = |sugg: &str| {
                if let Some(name) = self.tcx.hir_maybe_get_struct_pattern_shorthand_field(expr) {
                    format!(": {}{}", name, sugg)
                } else {
                    sugg.to_string()
                }
            };

            // FIXME: This could/should be extended to suggest `as_mut` and `as_deref_mut`,
            // but those checks need to be a bit more delicate and the benefit is diminishing.
            if self.can_eq(self.param_env, found_ty_inner, peeled) && error_tys_equate_as_ref {
                let sugg = prefix_wrap(".as_ref()");
                err.subdiagnostic(errors::SuggestConvertViaMethod {
                    span: expr.span.shrink_to_hi(),
                    sugg,
                    expected,
                    found,
                    borrow_removal_span,
                });
                return true;
            } else if let ty::Ref(_, peeled_found_ty, _) = found_ty_inner.kind()
                && let ty::Adt(adt, _) = peeled_found_ty.peel_refs().kind()
                && self.tcx.is_lang_item(adt.did(), LangItem::String)
                && peeled.is_str()
                // `Result::map`, conversely, does not take ref of the error type.
                && error_tys.is_none_or(|(found, expected)| {
                    self.can_eq(self.param_env, found, expected)
                })
            {
                let sugg = prefix_wrap(".map(|x| x.as_str())");
                err.span_suggestion_verbose(
                    expr.span.shrink_to_hi(),
                    fluent::hir_typeck_convert_to_str,
                    sugg,
                    Applicability::MachineApplicable,
                );
                return true;
            } else {
                if !error_tys_equate_as_ref {
                    return false;
                }
                let mut steps = self.autoderef(expr.span, found_ty_inner).silence_errors();
                if let Some((deref_ty, _)) = steps.nth(1)
                    && self.can_eq(self.param_env, deref_ty, peeled)
                {
                    let sugg = prefix_wrap(".as_deref()");
                    err.subdiagnostic(errors::SuggestConvertViaMethod {
                        span: expr.span.shrink_to_hi(),
                        sugg,
                        expected,
                        found,
                        borrow_removal_span,
                    });
                    return true;
                }
                for (deref_ty, n_step) in steps {
                    if self.can_eq(self.param_env, deref_ty, peeled) {
                        let explicit_deref = "*".repeat(n_step);
                        let sugg = prefix_wrap(&format!(".map(|v| &{explicit_deref}v)"));
                        err.subdiagnostic(errors::SuggestConvertViaMethod {
                            span: expr.span.shrink_to_hi(),
                            sugg,
                            expected,
                            found,
                            borrow_removal_span,
                        });
                        return true;
                    }
                }
            }
        }

        false
    }

    /// If `ty` is `Option<T>`, returns `T, T, None`.
    /// If `ty` is `Result<T, E>`, returns `T, T, Some(E, E)`.
    /// Otherwise, returns `None`.
    fn deconstruct_option_or_result(
        &self,
        found_ty: Ty<'tcx>,
        expected_ty: Ty<'tcx>,
    ) -> Option<(Ty<'tcx>, Ty<'tcx>, Option<(Ty<'tcx>, Ty<'tcx>)>)> {
        let ty::Adt(found_adt, found_args) = found_ty.peel_refs().kind() else {
            return None;
        };
        let ty::Adt(expected_adt, expected_args) = expected_ty.kind() else {
            return None;
        };
        if self.tcx.is_diagnostic_item(sym::Option, found_adt.did())
            && self.tcx.is_diagnostic_item(sym::Option, expected_adt.did())
        {
            Some((found_args.type_at(0), expected_args.type_at(0), None))
        } else if self.tcx.is_diagnostic_item(sym::Result, found_adt.did())
            && self.tcx.is_diagnostic_item(sym::Result, expected_adt.did())
        {
            Some((
                found_args.type_at(0),
                expected_args.type_at(0),
                Some((found_args.type_at(1), expected_args.type_at(1))),
            ))
        } else {
            None
        }
    }

    /// When encountering the expected boxed value allocated in the stack, suggest allocating it
    /// in the heap by calling `Box::new()`.
    pub(in super::super) fn suggest_boxing_when_appropriate(
        &self,
        err: &mut Diag<'_>,
        span: Span,
        hir_id: HirId,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) -> bool {
        // Do not suggest `Box::new` in const context.
        if self.tcx.hir_is_inside_const_context(hir_id) || !expected.is_box() || found.is_box() {
            return false;
        }
        if self.may_coerce(Ty::new_box(self.tcx, found), expected) {
            let suggest_boxing = match found.kind() {
                ty::Tuple(tuple) if tuple.is_empty() => {
                    errors::SuggestBoxing::Unit { start: span.shrink_to_lo(), end: span }
                }
                ty::Coroutine(def_id, ..)
                    if matches!(
                        self.tcx.coroutine_kind(def_id),
                        Some(CoroutineKind::Desugared(
                            CoroutineDesugaring::Async,
                            CoroutineSource::Closure
                        ))
                    ) =>
                {
                    errors::SuggestBoxing::AsyncBody
                }
                _ if let Node::ExprField(expr_field) = self.tcx.parent_hir_node(hir_id)
                    && expr_field.is_shorthand =>
                {
                    errors::SuggestBoxing::ExprFieldShorthand {
                        start: span.shrink_to_lo(),
                        end: span.shrink_to_hi(),
                        ident: expr_field.ident,
                    }
                }
                _ => errors::SuggestBoxing::Other {
                    start: span.shrink_to_lo(),
                    end: span.shrink_to_hi(),
                },
            };
            err.subdiagnostic(suggest_boxing);

            true
        } else {
            false
        }
    }

    /// When encountering a closure that captures variables, where a FnPtr is expected,
    /// suggest a non-capturing closure
    pub(in super::super) fn suggest_no_capture_closure(
        &self,
        err: &mut Diag<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) -> bool {
        if let (ty::FnPtr(..), ty::Closure(def_id, _)) = (expected.kind(), found.kind()) {
            if let Some(upvars) = self.tcx.upvars_mentioned(*def_id) {
                // Report upto four upvars being captured to reduce the amount error messages
                // reported back to the user.
                let spans_and_labels = upvars
                    .iter()
                    .take(4)
                    .map(|(var_hir_id, upvar)| {
                        let var_name = self.tcx.hir_name(*var_hir_id).to_string();
                        let msg = format!("`{var_name}` captured here");
                        (upvar.span, msg)
                    })
                    .collect::<Vec<_>>();

                let mut multi_span: MultiSpan =
                    spans_and_labels.iter().map(|(sp, _)| *sp).collect::<Vec<_>>().into();
                for (sp, label) in spans_and_labels {
                    multi_span.push_span_label(sp, label);
                }
                err.span_note(
                    multi_span,
                    "closures can only be coerced to `fn` types if they do not capture any variables"
                );
                return true;
            }
        }
        false
    }

    /// When encountering an `impl Future` where `BoxFuture` is expected, suggest `Box::pin`.
    #[instrument(skip(self, err))]
    pub(in super::super) fn suggest_calling_boxed_future_when_appropriate(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) -> bool {
        // Handle #68197.

        if self.tcx.hir_is_inside_const_context(expr.hir_id) {
            // Do not suggest `Box::new` in const context.
            return false;
        }
        let pin_did = self.tcx.lang_items().pin_type();
        // This guards the `new_box` below.
        if pin_did.is_none() || self.tcx.lang_items().owned_box().is_none() {
            return false;
        }
        let box_found = Ty::new_box(self.tcx, found);
        let Some(pin_box_found) = Ty::new_lang_item(self.tcx, box_found, LangItem::Pin) else {
            return false;
        };
        let Some(pin_found) = Ty::new_lang_item(self.tcx, found, LangItem::Pin) else {
            return false;
        };
        match expected.kind() {
            ty::Adt(def, _) if Some(def.did()) == pin_did => {
                if self.may_coerce(pin_box_found, expected) {
                    debug!("can coerce {:?} to {:?}, suggesting Box::pin", pin_box_found, expected);
                    match found.kind() {
                        ty::Adt(def, _) if def.is_box() => {
                            err.help("use `Box::pin`");
                        }
                        _ => {
                            let prefix = if let Some(name) =
                                self.tcx.hir_maybe_get_struct_pattern_shorthand_field(expr)
                            {
                                format!("{}: ", name)
                            } else {
                                String::new()
                            };
                            let suggestion = vec![
                                (expr.span.shrink_to_lo(), format!("{prefix}Box::pin(")),
                                (expr.span.shrink_to_hi(), ")".to_string()),
                            ];
                            err.multipart_suggestion(
                                "you need to pin and box this expression",
                                suggestion,
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
                    true
                } else if self.may_coerce(pin_found, expected) {
                    match found.kind() {
                        ty::Adt(def, _) if def.is_box() => {
                            err.help("use `Box::pin`");
                            true
                        }
                        _ => false,
                    }
                } else {
                    false
                }
            }
            ty::Adt(def, _) if def.is_box() && self.may_coerce(box_found, expected) => {
                // Check if the parent expression is a call to Pin::new. If it
                // is and we were expecting a Box, ergo Pin<Box<expected>>, we
                // can suggest Box::pin.
                let Node::Expr(Expr { kind: ExprKind::Call(fn_name, _), .. }) =
                    self.tcx.parent_hir_node(expr.hir_id)
                else {
                    return false;
                };
                match fn_name.kind {
                    ExprKind::Path(QPath::TypeRelative(
                        hir::Ty {
                            kind: TyKind::Path(QPath::Resolved(_, Path { res: recv_ty, .. })),
                            ..
                        },
                        method,
                    )) if recv_ty.opt_def_id() == pin_did && method.ident.name == sym::new => {
                        err.span_suggestion(
                            fn_name.span,
                            "use `Box::pin` to pin and box this expression",
                            "Box::pin",
                            Applicability::MachineApplicable,
                        );
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// A common error is to forget to add a semicolon at the end of a block, e.g.,
    ///
    /// ```compile_fail,E0308
    /// # fn bar_that_returns_u32() -> u32 { 4 }
    /// fn foo() {
    ///     bar_that_returns_u32()
    /// }
    /// ```
    ///
    /// This routine checks if the return expression in a block would make sense on its own as a
    /// statement and the return type has been left as default or has been specified as `()`. If so,
    /// it suggests adding a semicolon.
    ///
    /// If the expression is the expression of a closure without block (`|| expr`), a
    /// block is needed to be added too (`|| { expr; }`). This is denoted by `needs_block`.
    pub(crate) fn suggest_missing_semicolon(
        &self,
        err: &mut Diag<'_>,
        expression: &'tcx hir::Expr<'tcx>,
        expected: Ty<'tcx>,
        needs_block: bool,
    ) {
        if expected.is_unit() {
            // `BlockTailExpression` only relevant if the tail expr would be
            // useful on its own.
            match expression.kind {
                ExprKind::Call(..)
                | ExprKind::MethodCall(..)
                | ExprKind::Loop(..)
                | ExprKind::If(..)
                | ExprKind::Match(..)
                | ExprKind::Block(..)
                    if expression.can_have_side_effects()
                        // If the expression is from an external macro, then do not suggest
                        // adding a semicolon, because there's nowhere to put it.
                        // See issue #81943.
                        && !expression.span.in_external_macro(self.tcx.sess.source_map()) =>
                {
                    if needs_block {
                        err.multipart_suggestion(
                            "consider using a semicolon here",
                            vec![
                                (expression.span.shrink_to_lo(), "{ ".to_owned()),
                                (expression.span.shrink_to_hi(), "; }".to_owned()),
                            ],
                            Applicability::MachineApplicable,
                        );
                    } else {
                        err.span_suggestion(
                            expression.span.shrink_to_hi(),
                            "consider using a semicolon here",
                            ";",
                            Applicability::MachineApplicable,
                        );
                    }
                }
                _ => (),
            }
        }
    }

    /// A possible error is to forget to add a return type that is needed:
    ///
    /// ```compile_fail,E0308
    /// # fn bar_that_returns_u32() -> u32 { 4 }
    /// fn foo() {
    ///     bar_that_returns_u32()
    /// }
    /// ```
    ///
    /// This routine checks if the return type is left as default, the method is not part of an
    /// `impl` block and that it isn't the `main` method. If so, it suggests setting the return
    /// type.
    #[instrument(level = "trace", skip(self, err))]
    pub(in super::super) fn suggest_missing_return_type(
        &self,
        err: &mut Diag<'_>,
        fn_decl: &hir::FnDecl<'tcx>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        fn_id: LocalDefId,
    ) -> bool {
        // Can't suggest `->` on a block-like coroutine
        if let Some(hir::CoroutineKind::Desugared(_, hir::CoroutineSource::Block)) =
            self.tcx.coroutine_kind(fn_id)
        {
            return false;
        }

        let found =
            self.resolve_numeric_literals_with_default(self.resolve_vars_if_possible(found));
        // Only suggest changing the return type for methods that
        // haven't set a return type at all (and aren't `fn main()`, impl or closure).
        match &fn_decl.output {
            // For closure with default returns, don't suggest adding return type
            &hir::FnRetTy::DefaultReturn(_) if self.tcx.is_closure_like(fn_id.to_def_id()) => {}
            &hir::FnRetTy::DefaultReturn(span) if expected.is_unit() => {
                if !self.can_add_return_type(fn_id) {
                    err.subdiagnostic(errors::ExpectedReturnTypeLabel::Unit { span });
                } else if let Some(found) = found.make_suggestable(self.tcx, false, None) {
                    err.subdiagnostic(errors::AddReturnTypeSuggestion::Add {
                        span,
                        found: found.to_string(),
                    });
                } else if let Some(sugg) = suggest_impl_trait(self, self.param_env, found) {
                    err.subdiagnostic(errors::AddReturnTypeSuggestion::Add { span, found: sugg });
                } else {
                    // FIXME: if `found` could be `impl Iterator` we should suggest that.
                    err.subdiagnostic(errors::AddReturnTypeSuggestion::MissingHere { span });
                }

                return true;
            }
            hir::FnRetTy::Return(hir_ty) => {
                if let hir::TyKind::OpaqueDef(op_ty, ..) = hir_ty.kind
                    // FIXME: account for RPITIT.
                    && let [hir::GenericBound::Trait(trait_ref)] = op_ty.bounds
                    && let Some(hir::PathSegment { args: Some(generic_args), .. }) =
                        trait_ref.trait_ref.path.segments.last()
                    && let [constraint] = generic_args.constraints
                    && let Some(ty) = constraint.ty()
                {
                    // Check if async function's return type was omitted.
                    // Don't emit suggestions if the found type is `impl Future<...>`.
                    debug!(?found);
                    if found.is_suggestable(self.tcx, false) {
                        if ty.span.is_empty() {
                            err.subdiagnostic(errors::AddReturnTypeSuggestion::Add {
                                span: ty.span,
                                found: found.to_string(),
                            });
                            return true;
                        } else {
                            err.subdiagnostic(errors::ExpectedReturnTypeLabel::Other {
                                span: ty.span,
                                expected,
                            });
                        }
                    }
                } else {
                    // Only point to return type if the expected type is the return type, as if they
                    // are not, the expectation must have been caused by something else.
                    debug!(?hir_ty, "return type");
                    let ty = self.lowerer().lower_ty(hir_ty);
                    debug!(?ty, "return type (lowered)");
                    debug!(?expected, "expected type");
                    let bound_vars =
                        self.tcx.late_bound_vars(self.tcx.local_def_id_to_hir_id(fn_id));
                    let ty = Binder::bind_with_vars(ty, bound_vars);
                    let ty = self.normalize(hir_ty.span, ty);
                    let ty = self.tcx.instantiate_bound_regions_with_erased(ty);
                    if self.may_coerce(expected, ty) {
                        err.subdiagnostic(errors::ExpectedReturnTypeLabel::Other {
                            span: hir_ty.span,
                            expected,
                        });
                        self.try_suggest_return_impl_trait(err, expected, found, fn_id);
                        self.try_note_caller_chooses_ty_for_ty_param(err, expected, found);
                        return true;
                    }
                }
            }
            _ => {}
        }
        false
    }

    /// Checks whether we can add a return type to a function.
    /// Assumes given function doesn't have a explicit return type.
    fn can_add_return_type(&self, fn_id: LocalDefId) -> bool {
        match self.tcx.hir_node_by_def_id(fn_id) {
            Node::Item(item) => {
                let (ident, _, _, _) = item.expect_fn();
                // This is less than ideal, it will not suggest a return type span on any
                // method called `main`, regardless of whether it is actually the entry point,
                // but it will still present it as the reason for the expected type.
                ident.name != sym::main
            }
            Node::ImplItem(item) => {
                // If it doesn't impl a trait, we can add a return type
                let Node::Item(&hir::Item {
                    kind: hir::ItemKind::Impl(&hir::Impl { of_trait, .. }),
                    ..
                }) = self.tcx.parent_hir_node(item.hir_id())
                else {
                    unreachable!();
                };

                of_trait.is_none()
            }
            _ => true,
        }
    }

    fn try_note_caller_chooses_ty_for_ty_param(
        &self,
        diag: &mut Diag<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) {
        // Only show the note if:
        // 1. `expected` ty is a type parameter;
        // 2. The `expected` type parameter does *not* occur in the return expression type. This can
        //    happen for e.g. `fn foo<T>(t: &T) -> T { t }`, where `expected` is `T` but `found` is
        //    `&T`. Saying "the caller chooses a type for `T` which can be different from `&T`" is
        //    "well duh" and is only confusing and not helpful.
        let ty::Param(expected_ty_as_param) = expected.kind() else {
            return;
        };

        if found.contains(expected) {
            return;
        }

        diag.subdiagnostic(errors::NoteCallerChoosesTyForTyParam {
            ty_param_name: expected_ty_as_param.name,
            found_ty: found,
        });
    }

    /// check whether the return type is a generic type with a trait bound
    /// only suggest this if the generic param is not present in the arguments
    /// if this is true, hint them towards changing the return type to `impl Trait`
    /// ```compile_fail,E0308
    /// fn cant_name_it<T: Fn() -> u32>() -> T {
    ///     || 3
    /// }
    /// ```
    fn try_suggest_return_impl_trait(
        &self,
        err: &mut Diag<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        fn_id: LocalDefId,
    ) {
        // Only apply the suggestion if:
        //  - the return type is a generic parameter
        //  - the generic param is not used as a fn param
        //  - the generic param has at least one bound
        //  - the generic param doesn't appear in any other bounds where it's not the Self type
        // Suggest:
        //  - Changing the return type to be `impl <all bounds>`

        debug!("try_suggest_return_impl_trait, expected = {:?}, found = {:?}", expected, found);

        let ty::Param(expected_ty_as_param) = expected.kind() else { return };

        let fn_node = self.tcx.hir_node_by_def_id(fn_id);

        let hir::Node::Item(hir::Item {
            kind:
                hir::ItemKind::Fn {
                    sig:
                        hir::FnSig {
                            decl: hir::FnDecl { inputs: fn_parameters, output: fn_return, .. },
                            ..
                        },
                    generics: hir::Generics { params, predicates, .. },
                    ..
                },
            ..
        }) = fn_node
        else {
            return;
        };

        if params.get(expected_ty_as_param.index as usize).is_none() {
            return;
        };

        // get all where BoundPredicates here, because they are used in two cases below
        let where_predicates = predicates
            .iter()
            .filter_map(|p| match p.kind {
                WherePredicateKind::BoundPredicate(hir::WhereBoundPredicate {
                    bounds,
                    bounded_ty,
                    ..
                }) => {
                    // FIXME: Maybe these calls to `lower_ty` can be removed (and the ones below)
                    let ty = self.lowerer().lower_ty(bounded_ty);
                    Some((ty, bounds))
                }
                _ => None,
            })
            .map(|(ty, bounds)| match ty.kind() {
                ty::Param(param_ty) if param_ty == expected_ty_as_param => Ok(Some(bounds)),
                // check whether there is any predicate that contains our `T`, like `Option<T>: Send`
                _ => match ty.contains(expected) {
                    true => Err(()),
                    false => Ok(None),
                },
            })
            .collect::<Result<Vec<_>, _>>();

        let Ok(where_predicates) = where_predicates else { return };

        // now get all predicates in the same types as the where bounds, so we can chain them
        let predicates_from_where =
            where_predicates.iter().flatten().flat_map(|bounds| bounds.iter());

        // extract all bounds from the source code using their spans
        let all_matching_bounds_strs = predicates_from_where
            .filter_map(|bound| match bound {
                GenericBound::Trait(_) => {
                    self.tcx.sess.source_map().span_to_snippet(bound.span()).ok()
                }
                _ => None,
            })
            .collect::<Vec<String>>();

        if all_matching_bounds_strs.len() == 0 {
            return;
        }

        let all_bounds_str = all_matching_bounds_strs.join(" + ");

        let ty_param_used_in_fn_params = fn_parameters.iter().any(|param| {
                let ty = self.lowerer().lower_ty( param);
                matches!(ty.kind(), ty::Param(fn_param_ty_param) if expected_ty_as_param == fn_param_ty_param)
            });

        if ty_param_used_in_fn_params {
            return;
        }

        err.span_suggestion(
            fn_return.span(),
            "consider using an impl return type",
            format!("impl {all_bounds_str}"),
            Applicability::MaybeIncorrect,
        );
    }

    pub(in super::super) fn suggest_missing_break_or_return_expr(
        &self,
        err: &mut Diag<'_>,
        expr: &'tcx hir::Expr<'tcx>,
        fn_decl: &hir::FnDecl<'tcx>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        id: HirId,
        fn_id: LocalDefId,
    ) {
        if !expected.is_unit() {
            return;
        }
        let found = self.resolve_vars_if_possible(found);

        let in_loop = self.is_loop(id)
            || self
                .tcx
                .hir_parent_iter(id)
                .take_while(|(_, node)| {
                    // look at parents until we find the first body owner
                    node.body_id().is_none()
                })
                .any(|(parent_id, _)| self.is_loop(parent_id));

        let in_local_statement = self.is_local_statement(id)
            || self
                .tcx
                .hir_parent_iter(id)
                .any(|(parent_id, _)| self.is_local_statement(parent_id));

        if in_loop && in_local_statement {
            err.multipart_suggestion(
                "you might have meant to break the loop with this value",
                vec![
                    (expr.span.shrink_to_lo(), "break ".to_string()),
                    (expr.span.shrink_to_hi(), ";".to_string()),
                ],
                Applicability::MaybeIncorrect,
            );
            return;
        }

        let scope = self.tcx.hir_parent_iter(id).find(|(_, node)| {
            matches!(
                node,
                Node::Expr(Expr { kind: ExprKind::Closure(..), .. })
                    | Node::Item(_)
                    | Node::TraitItem(_)
                    | Node::ImplItem(_)
            )
        });
        let in_closure =
            matches!(scope, Some((_, Node::Expr(Expr { kind: ExprKind::Closure(..), .. }))));

        let can_return = match fn_decl.output {
            hir::FnRetTy::Return(ty) => {
                let ty = self.lowerer().lower_ty(ty);
                let bound_vars = self.tcx.late_bound_vars(self.tcx.local_def_id_to_hir_id(fn_id));
                let ty = self
                    .tcx
                    .instantiate_bound_regions_with_erased(Binder::bind_with_vars(ty, bound_vars));
                let ty = match self.tcx.asyncness(fn_id) {
                    ty::Asyncness::Yes => {
                        self.err_ctxt().get_impl_future_output_ty(ty).unwrap_or_else(|| {
                            span_bug!(
                                fn_decl.output.span(),
                                "failed to get output type of async function"
                            )
                        })
                    }
                    ty::Asyncness::No => ty,
                };
                let ty = self.normalize(expr.span, ty);
                self.may_coerce(found, ty)
            }
            hir::FnRetTy::DefaultReturn(_) if in_closure => {
                self.ret_coercion.as_ref().is_some_and(|ret| {
                    let ret_ty = ret.borrow().expected_ty();
                    self.may_coerce(found, ret_ty)
                })
            }
            _ => false,
        };
        if can_return
            && let Some(span) = expr.span.find_ancestor_inside(
                self.tcx.hir_span_with_body(self.tcx.local_def_id_to_hir_id(fn_id)),
            )
        {
            // When the expr is in a match arm's body, we shouldn't add semicolon ';' at the end.
            // For example:
            // fn mismatch_types() -> i32 {
            //     match 1 {
            //         x => dbg!(x),
            //     }
            //     todo!()
            // }
            // -------------^^^^^^^-
            // Don't add semicolon `;` at the end of `dbg!(x)` expr
            fn is_in_arm<'tcx>(expr: &'tcx hir::Expr<'tcx>, tcx: TyCtxt<'tcx>) -> bool {
                for (_, node) in tcx.hir_parent_iter(expr.hir_id) {
                    match node {
                        hir::Node::Block(block) => {
                            if let Some(ret) = block.expr
                                && ret.hir_id == expr.hir_id
                            {
                                continue;
                            }
                        }
                        hir::Node::Arm(arm) => {
                            if let hir::ExprKind::Block(block, _) = arm.body.kind
                                && let Some(ret) = block.expr
                                && ret.hir_id == expr.hir_id
                            {
                                return true;
                            }
                        }
                        hir::Node::Expr(e) if let hir::ExprKind::Block(block, _) = e.kind => {
                            if let Some(ret) = block.expr
                                && ret.hir_id == expr.hir_id
                            {
                                continue;
                            }
                        }
                        _ => {
                            return false;
                        }
                    }
                }

                false
            }
            let mut suggs = vec![(span.shrink_to_lo(), "return ".to_string())];
            if !is_in_arm(expr, self.tcx) {
                suggs.push((span.shrink_to_hi(), ";".to_string()));
            }
            err.multipart_suggestion_verbose(
                "you might have meant to return this value",
                suggs,
                Applicability::MaybeIncorrect,
            );
        }
    }

    pub(in super::super) fn suggest_missing_parentheses(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
    ) -> bool {
        let sp = self.tcx.sess.source_map().start_point(expr.span).with_parent(None);
        if let Some(sp) = self.tcx.sess.psess.ambiguous_block_expr_parse.borrow().get(&sp) {
            // `{ 42 } &&x` (#61475) or `{ 42 } && if x { 1 } else { 0 }`
            err.subdiagnostic(ExprParenthesesNeeded::surrounding(*sp));
            true
        } else {
            false
        }
    }

    /// Given an expression type mismatch, peel any `&` expressions until we get to
    /// a block expression, and then suggest replacing the braces with square braces
    /// if it was possibly mistaken array syntax.
    pub(crate) fn suggest_block_to_brackets_peeling_refs(
        &self,
        diag: &mut Diag<'_>,
        mut expr: &hir::Expr<'_>,
        mut expr_ty: Ty<'tcx>,
        mut expected_ty: Ty<'tcx>,
    ) -> bool {
        loop {
            match (&expr.kind, expr_ty.kind(), expected_ty.kind()) {
                (
                    hir::ExprKind::AddrOf(_, _, inner_expr),
                    ty::Ref(_, inner_expr_ty, _),
                    ty::Ref(_, inner_expected_ty, _),
                ) => {
                    expr = *inner_expr;
                    expr_ty = *inner_expr_ty;
                    expected_ty = *inner_expected_ty;
                }
                (hir::ExprKind::Block(blk, _), _, _) => {
                    self.suggest_block_to_brackets(diag, *blk, expr_ty, expected_ty);
                    break true;
                }
                _ => break false,
            }
        }
    }

    pub(crate) fn suggest_clone_for_ref(
        &self,
        diag: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        expr_ty: Ty<'tcx>,
        expected_ty: Ty<'tcx>,
    ) -> bool {
        if let ty::Ref(_, inner_ty, hir::Mutability::Not) = expr_ty.kind()
            && let Some(clone_trait_def) = self.tcx.lang_items().clone_trait()
            && expected_ty == *inner_ty
            && self
                .infcx
                .type_implements_trait(
                    clone_trait_def,
                    [self.tcx.erase_regions(expected_ty)],
                    self.param_env,
                )
                .must_apply_modulo_regions()
        {
            let suggestion = match self.tcx.hir_maybe_get_struct_pattern_shorthand_field(expr) {
                Some(ident) => format!(": {ident}.clone()"),
                None => ".clone()".to_string(),
            };

            diag.span_suggestion_verbose(
                expr.span.shrink_to_hi(),
                "consider using clone here",
                suggestion,
                Applicability::MachineApplicable,
            );
            return true;
        }
        false
    }

    pub(crate) fn suggest_copied_cloned_or_as_ref(
        &self,
        diag: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        expr_ty: Ty<'tcx>,
        expected_ty: Ty<'tcx>,
    ) -> bool {
        let ty::Adt(adt_def, args) = expr_ty.kind() else {
            return false;
        };
        let ty::Adt(expected_adt_def, expected_args) = expected_ty.kind() else {
            return false;
        };
        if adt_def != expected_adt_def {
            return false;
        }

        if Some(adt_def.did()) == self.tcx.get_diagnostic_item(sym::Result)
            && self.can_eq(self.param_env, args.type_at(1), expected_args.type_at(1))
            || Some(adt_def.did()) == self.tcx.get_diagnostic_item(sym::Option)
        {
            let expr_inner_ty = args.type_at(0);
            let expected_inner_ty = expected_args.type_at(0);
            if let &ty::Ref(_, ty, _mutability) = expr_inner_ty.kind()
                && self.can_eq(self.param_env, ty, expected_inner_ty)
            {
                let def_path = self.tcx.def_path_str(adt_def.did());
                let span = expr.span.shrink_to_hi();
                let subdiag = if self.type_is_copy_modulo_regions(self.param_env, ty) {
                    errors::OptionResultRefMismatch::Copied { span, def_path }
                } else if self.type_is_clone_modulo_regions(self.param_env, ty) {
                    errors::OptionResultRefMismatch::Cloned { span, def_path }
                } else {
                    return false;
                };
                diag.subdiagnostic(subdiag);
                return true;
            }
        }

        false
    }

    pub(crate) fn suggest_into(
        &self,
        diag: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        expr_ty: Ty<'tcx>,
        expected_ty: Ty<'tcx>,
    ) -> bool {
        let expr = expr.peel_blocks();

        // We have better suggestions for scalar interconversions...
        if expr_ty.is_scalar() && expected_ty.is_scalar() {
            return false;
        }

        // Don't suggest turning a block into another type (e.g. `{}.into()`)
        if matches!(expr.kind, hir::ExprKind::Block(..)) {
            return false;
        }

        // We'll later suggest `.as_ref` when noting the type error,
        // so skip if we will suggest that instead.
        if self.err_ctxt().should_suggest_as_ref(expected_ty, expr_ty).is_some() {
            return false;
        }

        if let Some(into_def_id) = self.tcx.get_diagnostic_item(sym::Into)
            && self.predicate_must_hold_modulo_regions(&traits::Obligation::new(
                self.tcx,
                self.misc(expr.span),
                self.param_env,
                ty::TraitRef::new(self.tcx, into_def_id, [expr_ty, expected_ty]),
            ))
            && !expr
                .span
                .macro_backtrace()
                .any(|x| matches!(x.kind, ExpnKind::Macro(MacroKind::Attr | MacroKind::Derive, ..)))
        {
            let span = expr.span.find_oldest_ancestor_in_same_ctxt();

            let mut sugg = if expr.precedence() >= ExprPrecedence::Unambiguous {
                vec![(span.shrink_to_hi(), ".into()".to_owned())]
            } else {
                vec![
                    (span.shrink_to_lo(), "(".to_owned()),
                    (span.shrink_to_hi(), ").into()".to_owned()),
                ]
            };
            if let Some(name) = self.tcx.hir_maybe_get_struct_pattern_shorthand_field(expr) {
                sugg.insert(0, (expr.span.shrink_to_lo(), format!("{}: ", name)));
            }
            diag.multipart_suggestion(
                    format!("call `Into::into` on this expression to convert `{expr_ty}` into `{expected_ty}`"),
                    sugg,
                    Applicability::MaybeIncorrect
                );
            return true;
        }

        false
    }

    /// When expecting a `bool` and finding an `Option`, suggests using `let Some(..)` or `.is_some()`
    pub(crate) fn suggest_option_to_bool(
        &self,
        diag: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        expr_ty: Ty<'tcx>,
        expected_ty: Ty<'tcx>,
    ) -> bool {
        if !expected_ty.is_bool() {
            return false;
        }

        let ty::Adt(def, _) = expr_ty.peel_refs().kind() else {
            return false;
        };
        if !self.tcx.is_diagnostic_item(sym::Option, def.did()) {
            return false;
        }

        let cond_parent = self.tcx.hir_parent_iter(expr.hir_id).find(|(_, node)| {
            !matches!(node, hir::Node::Expr(hir::Expr { kind: hir::ExprKind::Binary(op, _, _), .. }) if op.node == hir::BinOpKind::And)
        });
        // Don't suggest:
        //     `let Some(_) = a.is_some() && b`
        //                     ++++++++++
        // since the user probably just misunderstood how `let else`
        // and `&&` work together.
        if let Some((_, hir::Node::LetStmt(local))) = cond_parent
            && let hir::PatKind::Expr(PatExpr { kind: PatExprKind::Path(qpath), .. })
            | hir::PatKind::TupleStruct(qpath, _, _) = &local.pat.kind
            && let hir::QPath::Resolved(None, path) = qpath
            && let Some(did) = path
                .res
                .opt_def_id()
                .and_then(|did| self.tcx.opt_parent(did))
                .and_then(|did| self.tcx.opt_parent(did))
            && self.tcx.is_diagnostic_item(sym::Option, did)
        {
            return false;
        }

        let suggestion = match self.tcx.hir_maybe_get_struct_pattern_shorthand_field(expr) {
            Some(ident) => format!(": {ident}.is_some()"),
            None => ".is_some()".to_string(),
        };

        diag.span_suggestion_verbose(
            expr.span.shrink_to_hi(),
            "use `Option::is_some` to test if the `Option` has a value",
            suggestion,
            Applicability::MachineApplicable,
        );
        true
    }

    // Suggest to change `Option<&Vec<T>>::unwrap_or(&[])` to `Option::map_or(&[], |v| v)`.
    #[instrument(level = "trace", skip(self, err, provided_expr))]
    pub(crate) fn suggest_deref_unwrap_or(
        &self,
        err: &mut Diag<'_>,
        callee_ty: Option<Ty<'tcx>>,
        call_ident: Option<Ident>,
        expected_ty: Ty<'tcx>,
        provided_ty: Ty<'tcx>,
        provided_expr: &Expr<'tcx>,
        is_method: bool,
    ) {
        if !is_method {
            return;
        }
        let Some(callee_ty) = callee_ty else {
            return;
        };
        let ty::Adt(callee_adt, _) = callee_ty.peel_refs().kind() else {
            return;
        };
        let adt_name = if self.tcx.is_diagnostic_item(sym::Option, callee_adt.did()) {
            "Option"
        } else if self.tcx.is_diagnostic_item(sym::Result, callee_adt.did()) {
            "Result"
        } else {
            return;
        };

        let Some(call_ident) = call_ident else {
            return;
        };
        if call_ident.name != sym::unwrap_or {
            return;
        }

        let ty::Ref(_, peeled, _mutability) = provided_ty.kind() else {
            return;
        };

        // NOTE: Can we reuse `suggest_deref_or_ref`?

        // Create an dummy type `&[_]` so that both &[] and `&Vec<T>` can coerce to it.
        let dummy_ty = if let ty::Array(elem_ty, size) = peeled.kind()
            && let ty::Infer(_) = elem_ty.kind()
            && self
                .try_structurally_resolve_const(provided_expr.span, *size)
                .try_to_target_usize(self.tcx)
                == Some(0)
        {
            let slice = Ty::new_slice(self.tcx, *elem_ty);
            Ty::new_imm_ref(self.tcx, self.tcx.lifetimes.re_static, slice)
        } else {
            provided_ty
        };

        if !self.may_coerce(expected_ty, dummy_ty) {
            return;
        }
        let msg = format!("use `{adt_name}::map_or` to deref inner value of `{adt_name}`");
        err.multipart_suggestion_verbose(
            msg,
            vec![
                (call_ident.span, "map_or".to_owned()),
                (provided_expr.span.shrink_to_hi(), ", |v| v".to_owned()),
            ],
            Applicability::MachineApplicable,
        );
    }

    /// Suggest wrapping the block in square brackets instead of curly braces
    /// in case the block was mistaken array syntax, e.g. `{ 1 }` -> `[ 1 ]`.
    pub(crate) fn suggest_block_to_brackets(
        &self,
        diag: &mut Diag<'_>,
        blk: &hir::Block<'_>,
        blk_ty: Ty<'tcx>,
        expected_ty: Ty<'tcx>,
    ) {
        if let ty::Slice(elem_ty) | ty::Array(elem_ty, _) = expected_ty.kind() {
            if self.may_coerce(blk_ty, *elem_ty)
                && blk.stmts.is_empty()
                && blk.rules == hir::BlockCheckMode::DefaultBlock
                && let source_map = self.tcx.sess.source_map()
                && let Ok(snippet) = source_map.span_to_snippet(blk.span)
                && snippet.starts_with('{')
                && snippet.ends_with('}')
            {
                diag.multipart_suggestion_verbose(
                    "to create an array, use square brackets instead of curly braces",
                    vec![
                        (
                            blk.span
                                .shrink_to_lo()
                                .with_hi(rustc_span::BytePos(blk.span.lo().0 + 1)),
                            "[".to_string(),
                        ),
                        (
                            blk.span
                                .shrink_to_hi()
                                .with_lo(rustc_span::BytePos(blk.span.hi().0 - 1)),
                            "]".to_string(),
                        ),
                    ],
                    Applicability::MachineApplicable,
                );
            }
        }
    }

    #[instrument(skip(self, err))]
    pub(crate) fn suggest_floating_point_literal(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        expected_ty: Ty<'tcx>,
    ) -> bool {
        if !expected_ty.is_floating_point() {
            return false;
        }
        match expr.kind {
            ExprKind::Struct(QPath::LangItem(LangItem::Range, ..), [start, end], _) => {
                err.span_suggestion_verbose(
                    start.span.shrink_to_hi().with_hi(end.span.lo()),
                    "remove the unnecessary `.` operator for a floating point literal",
                    '.',
                    Applicability::MaybeIncorrect,
                );
                true
            }
            ExprKind::Struct(QPath::LangItem(LangItem::RangeFrom, ..), [start], _) => {
                err.span_suggestion_verbose(
                    expr.span.with_lo(start.span.hi()),
                    "remove the unnecessary `.` operator for a floating point literal",
                    '.',
                    Applicability::MaybeIncorrect,
                );
                true
            }
            ExprKind::Struct(QPath::LangItem(LangItem::RangeTo, ..), [end], _) => {
                err.span_suggestion_verbose(
                    expr.span.until(end.span),
                    "remove the unnecessary `.` operator and add an integer part for a floating point literal",
                    "0.",
                    Applicability::MaybeIncorrect,
                );
                true
            }
            ExprKind::Lit(Spanned {
                node: rustc_ast::LitKind::Int(lit, rustc_ast::LitIntType::Unsuffixed),
                span,
            }) => {
                let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(*span) else {
                    return false;
                };
                if !(snippet.starts_with("0x") || snippet.starts_with("0X")) {
                    return false;
                }
                if snippet.len() <= 5 || !snippet.is_char_boundary(snippet.len() - 3) {
                    return false;
                }
                let (_, suffix) = snippet.split_at(snippet.len() - 3);
                let value = match suffix {
                    "f32" => (lit.get() - 0xf32) / (16 * 16 * 16),
                    "f64" => (lit.get() - 0xf64) / (16 * 16 * 16),
                    _ => return false,
                };
                err.span_suggestions(
                    expr.span,
                    "rewrite this as a decimal floating point literal, or use `as` to turn a hex literal into a float",
                    [format!("0x{value:X} as {suffix}"), format!("{value}_{suffix}")],
                    Applicability::MaybeIncorrect,
                );
                true
            }
            _ => false,
        }
    }

    /// Suggest providing `std::ptr::null()` or `std::ptr::null_mut()` if they
    /// pass in a literal 0 to an raw pointer.
    #[instrument(skip(self, err))]
    pub(crate) fn suggest_null_ptr_for_literal_zero_given_to_ptr_arg(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        expected_ty: Ty<'tcx>,
    ) -> bool {
        // Expected type needs to be a raw pointer.
        let ty::RawPtr(_, mutbl) = expected_ty.kind() else {
            return false;
        };

        // Provided expression needs to be a literal `0`.
        let ExprKind::Lit(Spanned { node: rustc_ast::LitKind::Int(Pu128(0), _), span }) = expr.kind
        else {
            return false;
        };

        // We need to find a null pointer symbol to suggest
        let null_sym = match mutbl {
            hir::Mutability::Not => sym::ptr_null,
            hir::Mutability::Mut => sym::ptr_null_mut,
        };
        let Some(null_did) = self.tcx.get_diagnostic_item(null_sym) else {
            return false;
        };
        let null_path_str = with_no_trimmed_paths!(self.tcx.def_path_str(null_did));

        // We have satisfied all requirements to provide a suggestion. Emit it.
        err.span_suggestion(
            *span,
            format!("if you meant to create a null pointer, use `{null_path_str}()`"),
            null_path_str + "()",
            Applicability::MachineApplicable,
        );

        true
    }

    pub(crate) fn suggest_associated_const(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'tcx>,
        expected_ty: Ty<'tcx>,
    ) -> bool {
        let Some((DefKind::AssocFn, old_def_id)) =
            self.typeck_results.borrow().type_dependent_def(expr.hir_id)
        else {
            return false;
        };
        let old_item_name = self.tcx.item_name(old_def_id);
        let capitalized_name = Symbol::intern(&old_item_name.as_str().to_uppercase());
        if old_item_name == capitalized_name {
            return false;
        }
        let (item, segment) = match expr.kind {
            hir::ExprKind::Path(QPath::Resolved(
                Some(ty),
                hir::Path { segments: [segment], .. },
            ))
            | hir::ExprKind::Path(QPath::TypeRelative(ty, segment)) => {
                if let Some(self_ty) = self.typeck_results.borrow().node_type_opt(ty.hir_id)
                    && let Ok(pick) = self.probe_for_name(
                        Mode::Path,
                        Ident::new(capitalized_name, segment.ident.span),
                        Some(expected_ty),
                        IsSuggestion(true),
                        self_ty,
                        expr.hir_id,
                        ProbeScope::TraitsInScope,
                    )
                {
                    (pick.item, segment)
                } else {
                    return false;
                }
            }
            hir::ExprKind::Path(QPath::Resolved(
                None,
                hir::Path { segments: [.., segment], .. },
            )) => {
                // we resolved through some path that doesn't end in the item name,
                // better not do a bad suggestion by accident.
                if old_item_name != segment.ident.name {
                    return false;
                }
                if let Some(item) = self
                    .tcx
                    .associated_items(self.tcx.parent(old_def_id))
                    .filter_by_name_unhygienic(capitalized_name)
                    .next()
                {
                    (*item, segment)
                } else {
                    return false;
                }
            }
            _ => return false,
        };
        if item.def_id == old_def_id || self.tcx.def_kind(item.def_id) != DefKind::AssocConst {
            // Same item
            return false;
        }
        let item_ty = self.tcx.type_of(item.def_id).instantiate_identity();
        // FIXME(compiler-errors): This check is *so* rudimentary
        if item_ty.has_param() {
            return false;
        }
        if self.may_coerce(item_ty, expected_ty) {
            err.span_suggestion_verbose(
                segment.ident.span,
                format!("try referring to the associated const `{capitalized_name}` instead",),
                capitalized_name,
                Applicability::MachineApplicable,
            );
            true
        } else {
            false
        }
    }

    fn is_loop(&self, id: HirId) -> bool {
        let node = self.tcx.hir_node(id);
        matches!(node, Node::Expr(Expr { kind: ExprKind::Loop(..), .. }))
    }

    fn is_local_statement(&self, id: HirId) -> bool {
        let node = self.tcx.hir_node(id);
        matches!(node, Node::Stmt(Stmt { kind: StmtKind::Let(..), .. }))
    }

    /// Suggest that `&T` was cloned instead of `T` because `T` does not implement `Clone`,
    /// which is a side-effect of autoref.
    pub(crate) fn note_type_is_not_clone(
        &self,
        diag: &mut Diag<'_>,
        expected_ty: Ty<'tcx>,
        found_ty: Ty<'tcx>,
        expr: &hir::Expr<'_>,
    ) {
        // When `expr` is `x` in something like `let x = foo.clone(); x`, need to recurse up to get
        // `foo` and `clone`.
        let expr = self.note_type_is_not_clone_inner_expr(expr);

        // If we've recursed to an `expr` of `foo.clone()`, get `foo` and `clone`.
        let hir::ExprKind::MethodCall(segment, callee_expr, &[], _) = expr.kind else {
            return;
        };

        let Some(clone_trait_did) = self.tcx.lang_items().clone_trait() else {
            return;
        };
        let ty::Ref(_, pointee_ty, _) = found_ty.kind() else { return };
        let results = self.typeck_results.borrow();
        // First, look for a `Clone::clone` call
        if segment.ident.name == sym::clone
            && results.type_dependent_def_id(expr.hir_id).is_some_and(|did| {
                    let assoc_item = self.tcx.associated_item(did);
                    assoc_item.container == ty::AssocItemContainer::Trait
                        && assoc_item.container_id(self.tcx) == clone_trait_did
                })
            // If that clone call hasn't already dereferenced the self type (i.e. don't give this
            // diagnostic in cases where we have `(&&T).clone()` and we expect `T`).
            && !results.expr_adjustments(callee_expr).iter().any(|adj| matches!(adj.kind, ty::adjustment::Adjust::Deref(..)))
            // Check that we're in fact trying to clone into the expected type
            && self.may_coerce(*pointee_ty, expected_ty)
            && let trait_ref = ty::TraitRef::new(self.tcx, clone_trait_did, [expected_ty])
            // And the expected type doesn't implement `Clone`
            && !self.predicate_must_hold_considering_regions(&traits::Obligation::new(
                self.tcx,
                traits::ObligationCause::dummy(),
                self.param_env,
                trait_ref,
            ))
        {
            diag.span_note(
                callee_expr.span,
                format!(
                    "`{expected_ty}` does not implement `Clone`, so `{found_ty}` was cloned instead"
                ),
            );
            let owner = self.tcx.hir_enclosing_body_owner(expr.hir_id);
            if let ty::Param(param) = expected_ty.kind()
                && let Some(generics) = self.tcx.hir_get_generics(owner)
            {
                suggest_constraining_type_params(
                    self.tcx,
                    generics,
                    diag,
                    vec![(param.name.as_str(), "Clone", Some(clone_trait_did))].into_iter(),
                    None,
                );
            } else {
                if let Some(errors) =
                    self.type_implements_trait_shallow(clone_trait_did, expected_ty, self.param_env)
                {
                    match &errors[..] {
                        [] => {}
                        [error] => {
                            diag.help(format!(
                                "`Clone` is not implemented because the trait bound `{}` is \
                                 not satisfied",
                                error.obligation.predicate,
                            ));
                        }
                        _ => {
                            diag.help(format!(
                                "`Clone` is not implemented because the following trait bounds \
                                 could not be satisfied: {}",
                                listify(&errors, |e| format!("`{}`", e.obligation.predicate))
                                    .unwrap(),
                            ));
                        }
                    }
                    for error in errors {
                        if let traits::FulfillmentErrorCode::Select(
                            traits::SelectionError::Unimplemented,
                        ) = error.code
                            && let ty::PredicateKind::Clause(ty::ClauseKind::Trait(pred)) =
                                error.obligation.predicate.kind().skip_binder()
                        {
                            self.infcx.err_ctxt().suggest_derive(
                                &error.obligation,
                                diag,
                                error.obligation.predicate.kind().rebind(pred),
                            );
                        }
                    }
                }
                self.suggest_derive(diag, &[(trait_ref.upcast(self.tcx), None, None)]);
            }
        }
    }

    /// Given a type mismatch error caused by `&T` being cloned instead of `T`, and
    /// the `expr` as the source of this type mismatch, try to find the method call
    /// as the source of this error and return that instead. Otherwise, return the
    /// original expression.
    fn note_type_is_not_clone_inner_expr<'b>(
        &'b self,
        expr: &'b hir::Expr<'b>,
    ) -> &'b hir::Expr<'b> {
        match expr.peel_blocks().kind {
            hir::ExprKind::Path(hir::QPath::Resolved(
                None,
                hir::Path { segments: [_], res: crate::Res::Local(binding), .. },
            )) => {
                let hir::Node::Pat(hir::Pat { hir_id, .. }) = self.tcx.hir_node(*binding) else {
                    return expr;
                };

                match self.tcx.parent_hir_node(*hir_id) {
                    // foo.clone()
                    hir::Node::LetStmt(hir::LetStmt { init: Some(init), .. }) => {
                        self.note_type_is_not_clone_inner_expr(init)
                    }
                    // When `expr` is more complex like a tuple
                    hir::Node::Pat(hir::Pat {
                        hir_id: pat_hir_id,
                        kind: hir::PatKind::Tuple(pats, ..),
                        ..
                    }) => {
                        let hir::Node::LetStmt(hir::LetStmt { init: Some(init), .. }) =
                            self.tcx.parent_hir_node(*pat_hir_id)
                        else {
                            return expr;
                        };

                        match init.peel_blocks().kind {
                            ExprKind::Tup(init_tup) => {
                                if let Some(init) = pats
                                    .iter()
                                    .enumerate()
                                    .filter(|x| x.1.hir_id == *hir_id)
                                    .find_map(|(i, _)| init_tup.get(i))
                                {
                                    self.note_type_is_not_clone_inner_expr(init)
                                } else {
                                    expr
                                }
                            }
                            _ => expr,
                        }
                    }
                    _ => expr,
                }
            }
            // If we're calling into a closure that may not be typed recurse into that call. no need
            // to worry if it's a call to a typed function or closure as this would ne handled
            // previously.
            hir::ExprKind::Call(Expr { kind: call_expr_kind, .. }, _) => {
                if let hir::ExprKind::Path(hir::QPath::Resolved(None, call_expr_path)) =
                    call_expr_kind
                    && let hir::Path { segments: [_], res: crate::Res::Local(binding), .. } =
                        call_expr_path
                    && let hir::Node::Pat(hir::Pat { hir_id, .. }) = self.tcx.hir_node(*binding)
                    && let hir::Node::LetStmt(hir::LetStmt { init: Some(init), .. }) =
                        self.tcx.parent_hir_node(*hir_id)
                    && let Expr {
                        kind: hir::ExprKind::Closure(hir::Closure { body: body_id, .. }),
                        ..
                    } = init
                {
                    let hir::Body { value: body_expr, .. } = self.tcx.hir_body(*body_id);
                    self.note_type_is_not_clone_inner_expr(body_expr)
                } else {
                    expr
                }
            }
            _ => expr,
        }
    }

    pub(crate) fn is_field_suggestable(
        &self,
        field: &ty::FieldDef,
        hir_id: HirId,
        span: Span,
    ) -> bool {
        // The field must be visible in the containing module.
        field.vis.is_accessible_from(self.tcx.parent_module(hir_id), self.tcx)
            // The field must not be unstable.
            && !matches!(
                self.tcx.eval_stability(field.did, None, rustc_span::DUMMY_SP, None),
                rustc_middle::middle::stability::EvalResult::Deny { .. }
            )
            // If the field is from an external crate it must not be `doc(hidden)`.
            && (field.did.is_local() || !self.tcx.is_doc_hidden(field.did))
            // If the field is hygienic it must come from the same syntax context.
            && self.tcx.def_ident_span(field.did).unwrap().normalize_to_macros_2_0().eq_ctxt(span)
    }

    pub(crate) fn suggest_missing_unwrap_expect(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'tcx>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) -> bool {
        let ty::Adt(adt, args) = found.kind() else {
            return false;
        };
        let ret_ty_matches = |diagnostic_item| {
            let Some(sig) = self.body_fn_sig() else {
                return false;
            };
            let ty::Adt(kind, _) = sig.output().kind() else {
                return false;
            };
            self.tcx.is_diagnostic_item(diagnostic_item, kind.did())
        };

        // don't suggest anything like `Ok(ok_val).unwrap()` , `Some(some_val).unwrap()`,
        // `None.unwrap()` etc.
        let is_ctor = matches!(
            expr.kind,
            hir::ExprKind::Call(
                hir::Expr {
                    kind: hir::ExprKind::Path(hir::QPath::Resolved(
                        None,
                        hir::Path { res: Res::Def(hir::def::DefKind::Ctor(_, _), _), .. },
                    )),
                    ..
                },
                ..,
            ) | hir::ExprKind::Path(hir::QPath::Resolved(
                None,
                hir::Path { res: Res::Def(hir::def::DefKind::Ctor(_, _), _), .. },
            )),
        );

        let (article, kind, variant, sugg_operator) =
            if self.tcx.is_diagnostic_item(sym::Result, adt.did()) {
                ("a", "Result", "Err", ret_ty_matches(sym::Result))
            } else if self.tcx.is_diagnostic_item(sym::Option, adt.did()) {
                ("an", "Option", "None", ret_ty_matches(sym::Option))
            } else {
                return false;
            };
        if is_ctor || !self.may_coerce(args.type_at(0), expected) {
            return false;
        }

        let (msg, sugg) = if sugg_operator {
            (
                format!(
                    "use the `?` operator to extract the `{found}` value, propagating \
                            {article} `{kind}::{variant}` value to the caller"
                ),
                "?",
            )
        } else {
            (
                format!(
                    "consider using `{kind}::expect` to unwrap the `{found}` value, \
                                panicking if the value is {article} `{kind}::{variant}`"
                ),
                ".expect(\"REASON\")",
            )
        };

        let sugg = match self.tcx.hir_maybe_get_struct_pattern_shorthand_field(expr) {
            Some(ident) => format!(": {ident}{sugg}"),
            None => sugg.to_string(),
        };

        let span = expr.span.find_oldest_ancestor_in_same_ctxt();
        err.span_suggestion_verbose(span.shrink_to_hi(), msg, sugg, Applicability::HasPlaceholders);
        true
    }

    pub(crate) fn suggest_coercing_result_via_try_operator(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'tcx>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) -> bool {
        let returned = matches!(
            self.tcx.parent_hir_node(expr.hir_id),
            hir::Node::Expr(hir::Expr { kind: hir::ExprKind::Ret(_), .. })
        ) || self.tcx.hir_get_fn_id_for_return_block(expr.hir_id).is_some();
        if returned
            && let ty::Adt(e, args_e) = expected.kind()
            && let ty::Adt(f, args_f) = found.kind()
            && e.did() == f.did()
            && Some(e.did()) == self.tcx.get_diagnostic_item(sym::Result)
            && let e_ok = args_e.type_at(0)
            && let f_ok = args_f.type_at(0)
            && self.infcx.can_eq(self.param_env, f_ok, e_ok)
            && let e_err = args_e.type_at(1)
            && let f_err = args_f.type_at(1)
            && self
                .infcx
                .type_implements_trait(
                    self.tcx.get_diagnostic_item(sym::Into).unwrap(),
                    [f_err, e_err],
                    self.param_env,
                )
                .must_apply_modulo_regions()
        {
            err.multipart_suggestion(
                "use `?` to coerce and return an appropriate `Err`, and wrap the resulting value \
                 in `Ok` so the expression remains of type `Result`",
                vec![
                    (expr.span.shrink_to_lo(), "Ok(".to_string()),
                    (expr.span.shrink_to_hi(), "?)".to_string()),
                ],
                Applicability::MaybeIncorrect,
            );
            return true;
        }
        false
    }

    // If the expr is a while or for loop and is the tail expr of its
    // enclosing body suggest returning a value right after it
    pub(crate) fn suggest_returning_value_after_loop(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'tcx>,
        expected: Ty<'tcx>,
    ) -> bool {
        let tcx = self.tcx;
        let enclosing_scope =
            tcx.hir_get_enclosing_scope(expr.hir_id).map(|hir_id| tcx.hir_node(hir_id));

        // Get tail expr of the enclosing block or body
        let tail_expr = if let Some(Node::Block(hir::Block { expr, .. })) = enclosing_scope
            && expr.is_some()
        {
            *expr
        } else {
            let body_def_id = tcx.hir_enclosing_body_owner(expr.hir_id);
            let body = tcx.hir_body_owned_by(body_def_id);

            // Get tail expr of the body
            match body.value.kind {
                // Regular function body etc.
                hir::ExprKind::Block(block, _) => block.expr,
                // Anon const body (there's no block in this case)
                hir::ExprKind::DropTemps(expr) => Some(expr),
                _ => None,
            }
        };

        let Some(tail_expr) = tail_expr else {
            return false; // Body doesn't have a tail expr we can compare with
        };

        // Get the loop expr within the tail expr
        let loop_expr_in_tail = match expr.kind {
            hir::ExprKind::Loop(_, _, hir::LoopSource::While, _) => tail_expr,
            hir::ExprKind::Loop(_, _, hir::LoopSource::ForLoop, _) => {
                match tail_expr.peel_drop_temps() {
                    Expr { kind: ExprKind::Match(_, [Arm { body, .. }], _), .. } => body,
                    _ => return false, // Not really a for loop
                }
            }
            _ => return false, // Not a while or a for loop
        };

        // If the expr is the loop expr in the tail
        // then make the suggestion
        if expr.hir_id == loop_expr_in_tail.hir_id {
            let span = expr.span;

            let (msg, suggestion) = if expected.is_never() {
                (
                    "consider adding a diverging expression here",
                    "`loop {}` or `panic!(\"...\")`".to_string(),
                )
            } else {
                ("consider returning a value here", format!("`{expected}` value"))
            };

            let src_map = tcx.sess.source_map();
            let suggestion = if src_map.is_multiline(expr.span) {
                let indentation = src_map.indentation_before(span).unwrap_or_else(String::new);
                format!("\n{indentation}/* {suggestion} */")
            } else {
                // If the entire expr is on a single line
                // put the suggestion also on the same line
                format!(" /* {suggestion} */")
            };

            err.span_suggestion_verbose(
                span.shrink_to_hi(),
                msg,
                suggestion,
                Applicability::MaybeIncorrect,
            );

            true
        } else {
            false
        }
    }

    /// Suggest replacing comma with semicolon in incorrect repeat expressions
    /// like `["_", 10]` or `vec![String::new(), 10]`.
    pub(crate) fn suggest_semicolon_in_repeat_expr(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        expr_ty: Ty<'tcx>,
    ) -> bool {
        // Check if `expr` is contained in array of two elements
        if let hir::Node::Expr(array_expr) = self.tcx.parent_hir_node(expr.hir_id)
            && let hir::ExprKind::Array(elements) = array_expr.kind
            && let [first, second] = &elements[..]
            && second.hir_id == expr.hir_id
        {
            // Span between the two elements of the array
            let comma_span = first.span.between(second.span);

            // Check if `expr` is a constant value of type `usize`.
            // This can only detect const variable declarations and
            // calls to const functions.

            // Checking this here instead of rustc_hir::hir because
            // this check needs access to `self.tcx` but rustc_hir
            // has no access to `TyCtxt`.
            let expr_is_const_usize = expr_ty.is_usize()
                && match expr.kind {
                    ExprKind::Path(QPath::Resolved(
                        None,
                        Path { res: Res::Def(DefKind::Const, _), .. },
                    )) => true,
                    ExprKind::Call(
                        Expr {
                            kind:
                                ExprKind::Path(QPath::Resolved(
                                    None,
                                    Path { res: Res::Def(DefKind::Fn, fn_def_id), .. },
                                )),
                            ..
                        },
                        _,
                    ) => self.tcx.is_const_fn(*fn_def_id),
                    _ => false,
                };

            // Type of the first element is guaranteed to be checked
            // when execution reaches here because `mismatched types`
            // error occurs only when type of second element of array
            // is not the same as type of first element.
            let first_ty = self.typeck_results.borrow().expr_ty(first);

            // `array_expr` is from a macro `vec!["a", 10]` if
            // 1. array expression's span is imported from a macro
            // 2. first element of array implements `Clone` trait
            // 3. second element is an integer literal or is an expression of `usize` like type
            if self.tcx.sess.source_map().is_imported(array_expr.span)
                && self.type_is_clone_modulo_regions(self.param_env, first_ty)
                && (expr.is_size_lit() || expr_ty.is_usize_like())
            {
                err.subdiagnostic(errors::ReplaceCommaWithSemicolon {
                    comma_span,
                    descr: "a vector",
                });
                return true;
            }

            // `array_expr` is from an array `["a", 10]` if
            // 1. first element of array implements `Copy` trait
            // 2. second element is an integer literal or is a const value of type `usize`
            if self.type_is_copy_modulo_regions(self.param_env, first_ty)
                && (expr.is_size_lit() || expr_is_const_usize)
            {
                err.subdiagnostic(errors::ReplaceCommaWithSemicolon {
                    comma_span,
                    descr: "an array",
                });
                return true;
            }
        }
        false
    }

    /// If the expected type is an enum (Issue #55250) with any variants whose
    /// sole field is of the found type, suggest such variants. (Issue #42764)
    pub(crate) fn suggest_compatible_variants(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        expected: Ty<'tcx>,
        expr_ty: Ty<'tcx>,
    ) -> bool {
        if expr.span.in_external_macro(self.tcx.sess.source_map()) {
            return false;
        }
        if let ty::Adt(expected_adt, args) = expected.kind() {
            if let hir::ExprKind::Field(base, ident) = expr.kind {
                let base_ty = self.typeck_results.borrow().expr_ty(base);
                if self.can_eq(self.param_env, base_ty, expected)
                    && let Some(base_span) = base.span.find_ancestor_inside(expr.span)
                {
                    err.span_suggestion_verbose(
                        expr.span.with_lo(base_span.hi()),
                        format!("consider removing the tuple struct field `{ident}`"),
                        "",
                        Applicability::MaybeIncorrect,
                    );
                    return true;
                }
            }

            // If the expression is of type () and it's the return expression of a block,
            // we suggest adding a separate return expression instead.
            // (To avoid things like suggesting `Ok(while .. { .. })`.)
            if expr_ty.is_unit() {
                let mut id = expr.hir_id;
                let mut parent;

                // Unroll desugaring, to make sure this works for `for` loops etc.
                loop {
                    parent = self.tcx.parent_hir_id(id);
                    let parent_span = self.tcx.hir_span(parent);
                    if parent_span.find_ancestor_inside(expr.span).is_some() {
                        // The parent node is part of the same span, so is the result of the
                        // same expansion/desugaring and not the 'real' parent node.
                        id = parent;
                        continue;
                    }
                    break;
                }

                if let hir::Node::Block(&hir::Block { span: block_span, expr: Some(e), .. }) =
                    self.tcx.hir_node(parent)
                {
                    if e.hir_id == id {
                        if let Some(span) = expr.span.find_ancestor_inside(block_span) {
                            let return_suggestions = if self
                                .tcx
                                .is_diagnostic_item(sym::Result, expected_adt.did())
                            {
                                vec!["Ok(())"]
                            } else if self.tcx.is_diagnostic_item(sym::Option, expected_adt.did()) {
                                vec!["None", "Some(())"]
                            } else {
                                return false;
                            };
                            if let Some(indent) =
                                self.tcx.sess.source_map().indentation_before(span.shrink_to_lo())
                            {
                                // Add a semicolon, except after `}`.
                                let semicolon =
                                    match self.tcx.sess.source_map().span_to_snippet(span) {
                                        Ok(s) if s.ends_with('}') => "",
                                        _ => ";",
                                    };
                                err.span_suggestions(
                                    span.shrink_to_hi(),
                                    "try adding an expression at the end of the block",
                                    return_suggestions
                                        .into_iter()
                                        .map(|r| format!("{semicolon}\n{indent}{r}")),
                                    Applicability::MaybeIncorrect,
                                );
                            }
                            return true;
                        }
                    }
                }
            }

            let compatible_variants: Vec<(String, _, _, Option<String>)> = expected_adt
                .variants()
                .iter()
                .filter(|variant| {
                    variant.fields.len() == 1
                })
                .filter_map(|variant| {
                    let sole_field = &variant.single_field();

                    let field_is_local = sole_field.did.is_local();
                    let field_is_accessible =
                        sole_field.vis.is_accessible_from(expr.hir_id.owner.def_id, self.tcx)
                        // Skip suggestions for unstable public fields (for example `Pin::__pointer`)
                        && matches!(self.tcx.eval_stability(sole_field.did, None, expr.span, None), EvalResult::Allow | EvalResult::Unmarked);

                    if !field_is_local && !field_is_accessible {
                        return None;
                    }

                    let note_about_variant_field_privacy = (field_is_local && !field_is_accessible)
                        .then(|| " (its field is private, but it's local to this crate and its privacy can be changed)".to_string());

                    let sole_field_ty = sole_field.ty(self.tcx, args);
                    if self.may_coerce(expr_ty, sole_field_ty) {
                        let variant_path =
                            with_no_trimmed_paths!(self.tcx.def_path_str(variant.def_id));
                        // FIXME #56861: DRYer prelude filtering
                        if let Some(path) = variant_path.strip_prefix("std::prelude::")
                            && let Some((_, path)) = path.split_once("::")
                        {
                            return Some((path.to_string(), variant.ctor_kind(), sole_field.name, note_about_variant_field_privacy));
                        }
                        Some((variant_path, variant.ctor_kind(), sole_field.name, note_about_variant_field_privacy))
                    } else {
                        None
                    }
                })
                .collect();

            let suggestions_for = |variant: &_, ctor_kind, field_name| {
                let prefix = match self.tcx.hir_maybe_get_struct_pattern_shorthand_field(expr) {
                    Some(ident) => format!("{ident}: "),
                    None => String::new(),
                };

                let (open, close) = match ctor_kind {
                    Some(CtorKind::Fn) => ("(".to_owned(), ")"),
                    None => (format!(" {{ {field_name}: "), " }"),

                    Some(CtorKind::Const) => unreachable!("unit variants don't have fields"),
                };

                // Suggest constructor as deep into the block tree as possible.
                // This fixes https://github.com/rust-lang/rust/issues/101065,
                // and also just helps make the most minimal suggestions.
                let mut expr = expr;
                while let hir::ExprKind::Block(block, _) = &expr.kind
                    && let Some(expr_) = &block.expr
                {
                    expr = expr_
                }

                vec![
                    (expr.span.shrink_to_lo(), format!("{prefix}{variant}{open}")),
                    (expr.span.shrink_to_hi(), close.to_owned()),
                ]
            };

            match &compatible_variants[..] {
                [] => { /* No variants to format */ }
                [(variant, ctor_kind, field_name, note)] => {
                    // Just a single matching variant.
                    err.multipart_suggestion_verbose(
                        format!(
                            "try wrapping the expression in `{variant}`{note}",
                            note = note.as_deref().unwrap_or("")
                        ),
                        suggestions_for(&**variant, *ctor_kind, *field_name),
                        Applicability::MaybeIncorrect,
                    );
                    return true;
                }
                _ => {
                    // More than one matching variant.
                    err.multipart_suggestions(
                        format!(
                            "try wrapping the expression in a variant of `{}`",
                            self.tcx.def_path_str(expected_adt.did())
                        ),
                        compatible_variants.into_iter().map(
                            |(variant, ctor_kind, field_name, _)| {
                                suggestions_for(&variant, ctor_kind, field_name)
                            },
                        ),
                        Applicability::MaybeIncorrect,
                    );
                    return true;
                }
            }
        }

        false
    }

    pub(crate) fn suggest_non_zero_new_unwrap(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        expected: Ty<'tcx>,
        expr_ty: Ty<'tcx>,
    ) -> bool {
        let tcx = self.tcx;
        let (adt, args, unwrap) = match expected.kind() {
            // In case `Option<NonZero<T>>` is wanted, but `T` is provided, suggest calling `new`.
            ty::Adt(adt, args) if tcx.is_diagnostic_item(sym::Option, adt.did()) => {
                let nonzero_type = args.type_at(0); // Unwrap option type.
                let ty::Adt(adt, args) = nonzero_type.kind() else {
                    return false;
                };
                (adt, args, "")
            }
            // In case `NonZero<T>` is wanted but `T` is provided, also add `.unwrap()` to satisfy types.
            ty::Adt(adt, args) => (adt, args, ".unwrap()"),
            _ => return false,
        };

        if !self.tcx.is_diagnostic_item(sym::NonZero, adt.did()) {
            return false;
        }

        let int_type = args.type_at(0);
        if !self.may_coerce(expr_ty, int_type) {
            return false;
        }

        err.multipart_suggestion(
            format!("consider calling `{}::new`", sym::NonZero),
            vec![
                (expr.span.shrink_to_lo(), format!("{}::new(", sym::NonZero)),
                (expr.span.shrink_to_hi(), format!("){unwrap}")),
            ],
            Applicability::MaybeIncorrect,
        );

        true
    }

    /// Identify some cases where `as_ref()` would be appropriate and suggest it.
    ///
    /// Given the following code:
    /// ```compile_fail,E0308
    /// struct Foo;
    /// fn takes_ref(_: &Foo) {}
    /// let ref opt = Some(Foo);
    ///
    /// opt.map(|param| takes_ref(param));
    /// ```
    /// Suggest using `opt.as_ref().map(|param| takes_ref(param));` instead.
    ///
    /// It only checks for `Option` and `Result` and won't work with
    /// ```ignore (illustrative)
    /// opt.map(|param| { takes_ref(param) });
    /// ```
    fn can_use_as_ref(&self, expr: &hir::Expr<'_>) -> Option<(Vec<(Span, String)>, &'static str)> {
        let hir::ExprKind::Path(hir::QPath::Resolved(_, path)) = expr.kind else {
            return None;
        };

        let hir::def::Res::Local(local_id) = path.res else {
            return None;
        };

        let Node::Param(hir::Param { hir_id: param_hir_id, .. }) =
            self.tcx.parent_hir_node(local_id)
        else {
            return None;
        };

        let Node::Expr(hir::Expr {
            hir_id: expr_hir_id,
            kind: hir::ExprKind::Closure(hir::Closure { fn_decl: closure_fn_decl, .. }),
            ..
        }) = self.tcx.parent_hir_node(*param_hir_id)
        else {
            return None;
        };

        let hir = self.tcx.parent_hir_node(*expr_hir_id);
        let closure_params_len = closure_fn_decl.inputs.len();
        let (
            Node::Expr(hir::Expr {
                kind: hir::ExprKind::MethodCall(method_path, receiver, ..),
                ..
            }),
            1,
        ) = (hir, closure_params_len)
        else {
            return None;
        };

        let self_ty = self.typeck_results.borrow().expr_ty(receiver);
        let name = method_path.ident.name;
        let is_as_ref_able = match self_ty.peel_refs().kind() {
            ty::Adt(def, _) => {
                (self.tcx.is_diagnostic_item(sym::Option, def.did())
                    || self.tcx.is_diagnostic_item(sym::Result, def.did()))
                    && (name == sym::map || name == sym::and_then)
            }
            _ => false,
        };
        if is_as_ref_able {
            Some((
                vec![(method_path.ident.span.shrink_to_lo(), "as_ref().".to_string())],
                "consider using `as_ref` instead",
            ))
        } else {
            None
        }
    }

    /// This function is used to determine potential "simple" improvements or users' errors and
    /// provide them useful help. For example:
    ///
    /// ```compile_fail,E0308
    /// fn some_fn(s: &str) {}
    ///
    /// let x = "hey!".to_owned();
    /// some_fn(x); // error
    /// ```
    ///
    /// No need to find every potential function which could make a coercion to transform a
    /// `String` into a `&str` since a `&` would do the trick!
    ///
    /// In addition of this check, it also checks between references mutability state. If the
    /// expected is mutable but the provided isn't, maybe we could just say "Hey, try with
    /// `&mut`!".
    pub(crate) fn suggest_deref_or_ref(
        &self,
        expr: &hir::Expr<'tcx>,
        checked_ty: Ty<'tcx>,
        expected: Ty<'tcx>,
    ) -> Option<(
        Vec<(Span, String)>,
        String,
        Applicability,
        bool, /* verbose */
        bool, /* suggest `&` or `&mut` type annotation */
    )> {
        let sess = self.sess();
        let sp = expr.span;
        let sm = sess.source_map();

        // If the span is from an external macro, there's no suggestion we can make.
        if sp.in_external_macro(sm) {
            return None;
        }

        let replace_prefix = |s: &str, old: &str, new: &str| {
            s.strip_prefix(old).map(|stripped| new.to_string() + stripped)
        };

        // `ExprKind::DropTemps` is semantically irrelevant for these suggestions.
        let expr = expr.peel_drop_temps();

        match (&expr.kind, expected.kind(), checked_ty.kind()) {
            (_, &ty::Ref(_, exp, _), &ty::Ref(_, check, _)) => match (exp.kind(), check.kind()) {
                (&ty::Str, &ty::Array(arr, _) | &ty::Slice(arr)) if arr == self.tcx.types.u8 => {
                    if let hir::ExprKind::Lit(_) = expr.kind
                        && let Ok(src) = sm.span_to_snippet(sp)
                        && replace_prefix(&src, "b\"", "\"").is_some()
                    {
                        let pos = sp.lo() + BytePos(1);
                        return Some((
                            vec![(sp.with_hi(pos), String::new())],
                            "consider removing the leading `b`".to_string(),
                            Applicability::MachineApplicable,
                            true,
                            false,
                        ));
                    }
                }
                (&ty::Array(arr, _) | &ty::Slice(arr), &ty::Str) if arr == self.tcx.types.u8 => {
                    if let hir::ExprKind::Lit(_) = expr.kind
                        && let Ok(src) = sm.span_to_snippet(sp)
                        && replace_prefix(&src, "\"", "b\"").is_some()
                    {
                        return Some((
                            vec![(sp.shrink_to_lo(), "b".to_string())],
                            "consider adding a leading `b`".to_string(),
                            Applicability::MachineApplicable,
                            true,
                            false,
                        ));
                    }
                }
                _ => {}
            },
            (_, &ty::Ref(_, _, mutability), _) => {
                // Check if it can work when put into a ref. For example:
                //
                // ```
                // fn bar(x: &mut i32) {}
                //
                // let x = 0u32;
                // bar(&x); // error, expected &mut
                // ```
                let ref_ty = match mutability {
                    hir::Mutability::Mut => {
                        Ty::new_mut_ref(self.tcx, self.tcx.lifetimes.re_static, checked_ty)
                    }
                    hir::Mutability::Not => {
                        Ty::new_imm_ref(self.tcx, self.tcx.lifetimes.re_static, checked_ty)
                    }
                };
                if self.may_coerce(ref_ty, expected) {
                    let mut sugg_sp = sp;
                    if let hir::ExprKind::MethodCall(segment, receiver, args, _) = expr.kind {
                        let clone_trait =
                            self.tcx.require_lang_item(LangItem::Clone, Some(segment.ident.span));
                        if args.is_empty()
                            && self
                                .typeck_results
                                .borrow()
                                .type_dependent_def_id(expr.hir_id)
                                .is_some_and(|did| {
                                    let ai = self.tcx.associated_item(did);
                                    ai.trait_container(self.tcx) == Some(clone_trait)
                                })
                            && segment.ident.name == sym::clone
                        {
                            // If this expression had a clone call when suggesting borrowing
                            // we want to suggest removing it because it'd now be unnecessary.
                            sugg_sp = receiver.span;
                        }
                    }

                    if let hir::ExprKind::Unary(hir::UnOp::Deref, inner) = expr.kind
                        && let Some(1) = self.deref_steps_for_suggestion(expected, checked_ty)
                        && self.typeck_results.borrow().expr_ty(inner).is_ref()
                    {
                        // We have `*&T`, check if what was expected was `&T`.
                        // If so, we may want to suggest removing a `*`.
                        sugg_sp = sugg_sp.with_hi(inner.span.lo());
                        return Some((
                            vec![(sugg_sp, String::new())],
                            "consider removing deref here".to_string(),
                            Applicability::MachineApplicable,
                            true,
                            false,
                        ));
                    }

                    if let Some((sugg, msg)) = self.can_use_as_ref(expr) {
                        return Some((
                            sugg,
                            msg.to_string(),
                            Applicability::MachineApplicable,
                            true,
                            false,
                        ));
                    }

                    let prefix = match self.tcx.hir_maybe_get_struct_pattern_shorthand_field(expr) {
                        Some(ident) => format!("{ident}: "),
                        None => String::new(),
                    };

                    if let hir::Node::Expr(hir::Expr { kind: hir::ExprKind::Assign(..), .. }) =
                        self.tcx.parent_hir_node(expr.hir_id)
                    {
                        if mutability.is_mut() {
                            // Suppressing this diagnostic, we'll properly print it in `check_expr_assign`
                            return None;
                        }
                    }

                    let make_sugg = |expr: &Expr<'_>, span: Span, sugg: &str| {
                        if expr_needs_parens(expr) {
                            (
                                vec![
                                    (span.shrink_to_lo(), format!("{prefix}{sugg}(")),
                                    (span.shrink_to_hi(), ")".to_string()),
                                ],
                                false,
                            )
                        } else {
                            (vec![(span.shrink_to_lo(), format!("{prefix}{sugg}"))], true)
                        }
                    };

                    // Suggest dereferencing the lhs for expressions such as `&T <= T`
                    if let hir::Node::Expr(hir::Expr {
                        kind: hir::ExprKind::Binary(_, lhs, ..),
                        ..
                    }) = self.tcx.parent_hir_node(expr.hir_id)
                        && let &ty::Ref(..) = self.check_expr(lhs).kind()
                    {
                        let (sugg, verbose) = make_sugg(lhs, lhs.span, "*");

                        return Some((
                            sugg,
                            "consider dereferencing the borrow".to_string(),
                            Applicability::MachineApplicable,
                            verbose,
                            false,
                        ));
                    }

                    let sugg = mutability.ref_prefix_str();
                    let (sugg, verbose) = make_sugg(expr, sp, sugg);
                    return Some((
                        sugg,
                        format!("consider {}borrowing here", mutability.mutably_str()),
                        Applicability::MachineApplicable,
                        verbose,
                        false,
                    ));
                }
            }
            (hir::ExprKind::AddrOf(hir::BorrowKind::Ref, _, expr), _, &ty::Ref(_, checked, _))
                if self.can_eq(self.param_env, checked, expected) =>
            {
                let make_sugg = |start: Span, end: BytePos| {
                    // skip `(` for tuples such as `(c) = (&123)`.
                    // make sure we won't suggest like `(c) = 123)` which is incorrect.
                    let sp = sm
                        .span_extend_while(start.shrink_to_lo(), |c| c == '(' || c.is_whitespace())
                        .map_or(start, |s| s.shrink_to_hi());
                    Some((
                        vec![(sp.with_hi(end), String::new())],
                        "consider removing the borrow".to_string(),
                        Applicability::MachineApplicable,
                        true,
                        true,
                    ))
                };

                // We have `&T`, check if what was expected was `T`. If so,
                // we may want to suggest removing a `&`.
                if sm.is_imported(expr.span) {
                    // Go through the spans from which this span was expanded,
                    // and find the one that's pointing inside `sp`.
                    //
                    // E.g. for `&format!("")`, where we want the span to the
                    // `format!()` invocation instead of its expansion.
                    if let Some(call_span) =
                        iter::successors(Some(expr.span), |s| s.parent_callsite())
                            .find(|&s| sp.contains(s))
                        && sm.is_span_accessible(call_span)
                    {
                        return make_sugg(sp, call_span.lo());
                    }
                    return None;
                }
                if sp.contains(expr.span) && sm.is_span_accessible(expr.span) {
                    return make_sugg(sp, expr.span.lo());
                }
            }
            (_, &ty::RawPtr(ty_b, mutbl_b), &ty::Ref(_, ty_a, mutbl_a)) => {
                if let Some(steps) = self.deref_steps_for_suggestion(ty_a, ty_b)
                    // Only suggest valid if dereferencing needed.
                    && steps > 0
                    // The pointer type implements `Copy` trait so the suggestion is always valid.
                    && let Ok(src) = sm.span_to_snippet(sp)
                {
                    let derefs = "*".repeat(steps);
                    let old_prefix = mutbl_a.ref_prefix_str();
                    let new_prefix = mutbl_b.ref_prefix_str().to_owned() + &derefs;

                    let suggestion = replace_prefix(&src, old_prefix, &new_prefix).map(|_| {
                        // skip `&` or `&mut ` if both mutabilities are mutable
                        let lo = sp.lo()
                            + BytePos(min(old_prefix.len(), mutbl_b.ref_prefix_str().len()) as _);
                        // skip `&` or `&mut `
                        let hi = sp.lo() + BytePos(old_prefix.len() as _);
                        let sp = sp.with_lo(lo).with_hi(hi);

                        (
                            sp,
                            format!(
                                "{}{derefs}",
                                if mutbl_a != mutbl_b { mutbl_b.prefix_str() } else { "" }
                            ),
                            if mutbl_b <= mutbl_a {
                                Applicability::MachineApplicable
                            } else {
                                Applicability::MaybeIncorrect
                            },
                        )
                    });

                    if let Some((span, src, applicability)) = suggestion {
                        return Some((
                            vec![(span, src)],
                            "consider dereferencing".to_string(),
                            applicability,
                            true,
                            false,
                        ));
                    }
                }
            }
            _ if sp == expr.span => {
                if let Some(mut steps) = self.deref_steps_for_suggestion(checked_ty, expected) {
                    let mut expr = expr.peel_blocks();
                    let mut prefix_span = expr.span.shrink_to_lo();
                    let mut remove = String::new();

                    // Try peeling off any existing `&` and `&mut` to reach our target type
                    while steps > 0 {
                        if let hir::ExprKind::AddrOf(_, mutbl, inner) = expr.kind {
                            // If the expression has `&`, removing it would fix the error
                            prefix_span = prefix_span.with_hi(inner.span.lo());
                            expr = inner;
                            remove.push_str(mutbl.ref_prefix_str());
                            steps -= 1;
                        } else {
                            break;
                        }
                    }
                    // If we've reached our target type with just removing `&`, then just print now.
                    if steps == 0 && !remove.trim().is_empty() {
                        return Some((
                            vec![(prefix_span, String::new())],
                            format!("consider removing the `{}`", remove.trim()),
                            // Do not remove `&&` to get to bool, because it might be something like
                            // { a } && b, which we have a separate fixup suggestion that is more
                            // likely correct...
                            if remove.trim() == "&&" && expected == self.tcx.types.bool {
                                Applicability::MaybeIncorrect
                            } else {
                                Applicability::MachineApplicable
                            },
                            true,
                            false,
                        ));
                    }

                    // For this suggestion to make sense, the type would need to be `Copy`,
                    // or we have to be moving out of a `Box<T>`
                    if self.type_is_copy_modulo_regions(self.param_env, expected)
                        // FIXME(compiler-errors): We can actually do this if the checked_ty is
                        // `steps` layers of boxes, not just one, but this is easier and most likely.
                        || (checked_ty.is_box() && steps == 1)
                        // We can always deref a binop that takes its arguments by ref.
                        || matches!(
                            self.tcx.parent_hir_node(expr.hir_id),
                            hir::Node::Expr(hir::Expr { kind: hir::ExprKind::Binary(op, ..), .. })
                                if !op.node.is_by_value()
                        )
                    {
                        let deref_kind = if checked_ty.is_box() {
                            "unboxing the value"
                        } else if checked_ty.is_ref() {
                            "dereferencing the borrow"
                        } else {
                            "dereferencing the type"
                        };

                        // Suggest removing `&` if we have removed any, otherwise suggest just
                        // dereferencing the remaining number of steps.
                        let message = if remove.is_empty() {
                            format!("consider {deref_kind}")
                        } else {
                            format!(
                                "consider removing the `{}` and {} instead",
                                remove.trim(),
                                deref_kind
                            )
                        };

                        let prefix =
                            match self.tcx.hir_maybe_get_struct_pattern_shorthand_field(expr) {
                                Some(ident) => format!("{ident}: "),
                                None => String::new(),
                            };

                        let (span, suggestion) = if self.is_else_if_block(expr) {
                            // Don't suggest nonsense like `else *if`
                            return None;
                        } else if let Some(expr) = self.maybe_get_block_expr(expr) {
                            // prefix should be empty here..
                            (expr.span.shrink_to_lo(), "*".to_string())
                        } else {
                            (prefix_span, format!("{}{}", prefix, "*".repeat(steps)))
                        };
                        if suggestion.trim().is_empty() {
                            return None;
                        }

                        if expr_needs_parens(expr) {
                            return Some((
                                vec![
                                    (span, format!("{suggestion}(")),
                                    (expr.span.shrink_to_hi(), ")".to_string()),
                                ],
                                message,
                                Applicability::MachineApplicable,
                                true,
                                false,
                            ));
                        }

                        return Some((
                            vec![(span, suggestion)],
                            message,
                            Applicability::MachineApplicable,
                            true,
                            false,
                        ));
                    }
                }
            }
            _ => {}
        }
        None
    }

    /// Returns whether the given expression is an `else if`.
    fn is_else_if_block(&self, expr: &hir::Expr<'_>) -> bool {
        if let hir::ExprKind::If(..) = expr.kind {
            if let Node::Expr(hir::Expr {
                kind: hir::ExprKind::If(_, _, Some(else_expr)), ..
            }) = self.tcx.parent_hir_node(expr.hir_id)
            {
                return else_expr.hir_id == expr.hir_id;
            }
        }
        false
    }

    pub(crate) fn suggest_cast(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        checked_ty: Ty<'tcx>,
        expected_ty: Ty<'tcx>,
        expected_ty_expr: Option<&'tcx hir::Expr<'tcx>>,
    ) -> bool {
        if self.tcx.sess.source_map().is_imported(expr.span) {
            // Ignore if span is from within a macro.
            return false;
        }

        let span = if let hir::ExprKind::Lit(lit) = &expr.kind { lit.span } else { expr.span };
        let Ok(src) = self.tcx.sess.source_map().span_to_snippet(span) else {
            return false;
        };

        // If casting this expression to a given numeric type would be appropriate in case of a type
        // mismatch.
        //
        // We want to minimize the amount of casting operations that are suggested, as it can be a
        // lossy operation with potentially bad side effects, so we only suggest when encountering
        // an expression that indicates that the original type couldn't be directly changed.
        //
        // For now, don't suggest casting with `as`.
        let can_cast = false;

        let mut sugg = vec![];

        if let hir::Node::ExprField(field) = self.tcx.parent_hir_node(expr.hir_id) {
            // `expr` is a literal field for a struct, only suggest if appropriate
            if field.is_shorthand {
                // This is a field literal
                sugg.push((field.ident.span.shrink_to_lo(), format!("{}: ", field.ident)));
            } else {
                // Likely a field was meant, but this field wasn't found. Do not suggest anything.
                return false;
            }
        };

        if let hir::ExprKind::Call(path, args) = &expr.kind
            && let (hir::ExprKind::Path(hir::QPath::TypeRelative(base_ty, path_segment)), 1) =
                (&path.kind, args.len())
            // `expr` is a conversion like `u32::from(val)`, do not suggest anything (#63697).
            && let (hir::TyKind::Path(hir::QPath::Resolved(None, base_ty_path)), sym::from) =
                (&base_ty.kind, path_segment.ident.name)
        {
            if let Some(ident) = &base_ty_path.segments.iter().map(|s| s.ident).next() {
                match ident.name {
                    sym::i128
                    | sym::i64
                    | sym::i32
                    | sym::i16
                    | sym::i8
                    | sym::u128
                    | sym::u64
                    | sym::u32
                    | sym::u16
                    | sym::u8
                    | sym::isize
                    | sym::usize
                        if base_ty_path.segments.len() == 1 =>
                    {
                        return false;
                    }
                    _ => {}
                }
            }
        }

        let msg = format!(
            "you can convert {} `{}` to {} `{}`",
            checked_ty.kind().article(),
            checked_ty,
            expected_ty.kind().article(),
            expected_ty,
        );
        let cast_msg = format!(
            "you can cast {} `{}` to {} `{}`",
            checked_ty.kind().article(),
            checked_ty,
            expected_ty.kind().article(),
            expected_ty,
        );
        let lit_msg = format!(
            "change the type of the numeric literal from `{checked_ty}` to `{expected_ty}`",
        );

        let close_paren = if expr.precedence() < ExprPrecedence::Unambiguous {
            sugg.push((expr.span.shrink_to_lo(), "(".to_string()));
            ")"
        } else {
            ""
        };

        let mut cast_suggestion = sugg.clone();
        cast_suggestion.push((expr.span.shrink_to_hi(), format!("{close_paren} as {expected_ty}")));
        let mut into_suggestion = sugg.clone();
        into_suggestion.push((expr.span.shrink_to_hi(), format!("{close_paren}.into()")));
        let mut suffix_suggestion = sugg.clone();
        suffix_suggestion.push((
            if matches!(
                (expected_ty.kind(), checked_ty.kind()),
                (ty::Int(_) | ty::Uint(_), ty::Float(_))
            ) {
                // Remove fractional part from literal, for example `42.0f32` into `42`
                let src = src.trim_end_matches(&checked_ty.to_string());
                let len = src.split('.').next().unwrap().len();
                span.with_lo(span.lo() + BytePos(len as u32))
            } else {
                let len = src.trim_end_matches(&checked_ty.to_string()).len();
                span.with_lo(span.lo() + BytePos(len as u32))
            },
            if expr.precedence() < ExprPrecedence::Unambiguous {
                // Readd `)`
                format!("{expected_ty})")
            } else {
                expected_ty.to_string()
            },
        ));
        let literal_is_ty_suffixed = |expr: &hir::Expr<'_>| {
            if let hir::ExprKind::Lit(lit) = &expr.kind { lit.node.is_suffixed() } else { false }
        };
        let is_negative_int =
            |expr: &hir::Expr<'_>| matches!(expr.kind, hir::ExprKind::Unary(hir::UnOp::Neg, ..));
        let is_uint = |ty: Ty<'_>| matches!(ty.kind(), ty::Uint(..));

        let in_const_context = self.tcx.hir_is_inside_const_context(expr.hir_id);

        let suggest_fallible_into_or_lhs_from =
            |err: &mut Diag<'_>, exp_to_found_is_fallible: bool| {
                // If we know the expression the expected type is derived from, we might be able
                // to suggest a widening conversion rather than a narrowing one (which may
                // panic). For example, given x: u8 and y: u32, if we know the span of "x",
                //   x > y
                // can be given the suggestion "u32::from(x) > y" rather than
                // "x > y.try_into().unwrap()".
                let lhs_expr_and_src = expected_ty_expr.and_then(|expr| {
                    self.tcx
                        .sess
                        .source_map()
                        .span_to_snippet(expr.span)
                        .ok()
                        .map(|src| (expr, src))
                });
                let (msg, suggestion) = if let (Some((lhs_expr, lhs_src)), false) =
                    (lhs_expr_and_src, exp_to_found_is_fallible)
                {
                    let msg = format!(
                        "you can convert `{lhs_src}` from `{expected_ty}` to `{checked_ty}`, matching the type of `{src}`",
                    );
                    let suggestion = vec![
                        (lhs_expr.span.shrink_to_lo(), format!("{checked_ty}::from(")),
                        (lhs_expr.span.shrink_to_hi(), ")".to_string()),
                    ];
                    (msg, suggestion)
                } else {
                    let msg =
                        format!("{} and panic if the converted value doesn't fit", msg.clone());
                    let mut suggestion = sugg.clone();
                    suggestion.push((
                        expr.span.shrink_to_hi(),
                        format!("{close_paren}.try_into().unwrap()"),
                    ));
                    (msg, suggestion)
                };
                err.multipart_suggestion_verbose(msg, suggestion, Applicability::MachineApplicable);
            };

        let suggest_to_change_suffix_or_into =
            |err: &mut Diag<'_>, found_to_exp_is_fallible: bool, exp_to_found_is_fallible: bool| {
                let exp_is_lhs = expected_ty_expr.is_some_and(|e| self.tcx.hir_is_lhs(e.hir_id));

                if exp_is_lhs {
                    return;
                }

                let always_fallible = found_to_exp_is_fallible
                    && (exp_to_found_is_fallible || expected_ty_expr.is_none());
                let msg = if literal_is_ty_suffixed(expr) {
                    lit_msg.clone()
                } else if always_fallible && (is_negative_int(expr) && is_uint(expected_ty)) {
                    // We now know that converting either the lhs or rhs is fallible. Before we
                    // suggest a fallible conversion, check if the value can never fit in the
                    // expected type.
                    let msg = format!("`{src}` cannot fit into type `{expected_ty}`");
                    err.note(msg);
                    return;
                } else if in_const_context {
                    // Do not recommend `into` or `try_into` in const contexts.
                    return;
                } else if found_to_exp_is_fallible {
                    return suggest_fallible_into_or_lhs_from(err, exp_to_found_is_fallible);
                } else {
                    msg.clone()
                };
                let suggestion = if literal_is_ty_suffixed(expr) {
                    suffix_suggestion.clone()
                } else {
                    into_suggestion.clone()
                };
                err.multipart_suggestion_verbose(msg, suggestion, Applicability::MachineApplicable);
            };

        match (expected_ty.kind(), checked_ty.kind()) {
            (ty::Int(exp), ty::Int(found)) => {
                let (f2e_is_fallible, e2f_is_fallible) = match (exp.bit_width(), found.bit_width())
                {
                    (Some(exp), Some(found)) if exp < found => (true, false),
                    (Some(exp), Some(found)) if exp > found => (false, true),
                    (None, Some(8 | 16)) => (false, true),
                    (Some(8 | 16), None) => (true, false),
                    (None, _) | (_, None) => (true, true),
                    _ => (false, false),
                };
                suggest_to_change_suffix_or_into(err, f2e_is_fallible, e2f_is_fallible);
                true
            }
            (ty::Uint(exp), ty::Uint(found)) => {
                let (f2e_is_fallible, e2f_is_fallible) = match (exp.bit_width(), found.bit_width())
                {
                    (Some(exp), Some(found)) if exp < found => (true, false),
                    (Some(exp), Some(found)) if exp > found => (false, true),
                    (None, Some(8 | 16)) => (false, true),
                    (Some(8 | 16), None) => (true, false),
                    (None, _) | (_, None) => (true, true),
                    _ => (false, false),
                };
                suggest_to_change_suffix_or_into(err, f2e_is_fallible, e2f_is_fallible);
                true
            }
            (&ty::Int(exp), &ty::Uint(found)) => {
                let (f2e_is_fallible, e2f_is_fallible) = match (exp.bit_width(), found.bit_width())
                {
                    (Some(exp), Some(found)) if found < exp => (false, true),
                    (None, Some(8)) => (false, true),
                    _ => (true, true),
                };
                suggest_to_change_suffix_or_into(err, f2e_is_fallible, e2f_is_fallible);
                true
            }
            (&ty::Uint(exp), &ty::Int(found)) => {
                let (f2e_is_fallible, e2f_is_fallible) = match (exp.bit_width(), found.bit_width())
                {
                    (Some(exp), Some(found)) if found > exp => (true, false),
                    (Some(8), None) => (true, false),
                    _ => (true, true),
                };
                suggest_to_change_suffix_or_into(err, f2e_is_fallible, e2f_is_fallible);
                true
            }
            (ty::Float(exp), ty::Float(found)) => {
                if found.bit_width() < exp.bit_width() {
                    suggest_to_change_suffix_or_into(err, false, true);
                } else if literal_is_ty_suffixed(expr) {
                    err.multipart_suggestion_verbose(
                        lit_msg,
                        suffix_suggestion,
                        Applicability::MachineApplicable,
                    );
                } else if can_cast {
                    // Missing try_into implementation for `f64` to `f32`
                    err.multipart_suggestion_verbose(
                        format!("{cast_msg}, producing the closest possible value"),
                        cast_suggestion,
                        Applicability::MaybeIncorrect, // lossy conversion
                    );
                }
                true
            }
            (&ty::Uint(_) | &ty::Int(_), &ty::Float(_)) => {
                if literal_is_ty_suffixed(expr) {
                    err.multipart_suggestion_verbose(
                        lit_msg,
                        suffix_suggestion,
                        Applicability::MachineApplicable,
                    );
                } else if can_cast {
                    // Missing try_into implementation for `{float}` to `{integer}`
                    err.multipart_suggestion_verbose(
                        format!("{msg}, rounding the float towards zero"),
                        cast_suggestion,
                        Applicability::MaybeIncorrect, // lossy conversion
                    );
                }
                true
            }
            (ty::Float(exp), ty::Uint(found)) => {
                // if `found` is `None` (meaning found is `usize`), don't suggest `.into()`
                if exp.bit_width() > found.bit_width().unwrap_or(256) {
                    err.multipart_suggestion_verbose(
                        format!(
                            "{msg}, producing the floating point representation of the integer",
                        ),
                        into_suggestion,
                        Applicability::MachineApplicable,
                    );
                } else if literal_is_ty_suffixed(expr) {
                    err.multipart_suggestion_verbose(
                        lit_msg,
                        suffix_suggestion,
                        Applicability::MachineApplicable,
                    );
                } else {
                    // Missing try_into implementation for `{integer}` to `{float}`
                    err.multipart_suggestion_verbose(
                        format!(
                            "{cast_msg}, producing the floating point representation of the integer, \
                                 rounded if necessary",
                        ),
                        cast_suggestion,
                        Applicability::MaybeIncorrect, // lossy conversion
                    );
                }
                true
            }
            (ty::Float(exp), ty::Int(found)) => {
                // if `found` is `None` (meaning found is `isize`), don't suggest `.into()`
                if exp.bit_width() > found.bit_width().unwrap_or(256) {
                    err.multipart_suggestion_verbose(
                        format!(
                            "{}, producing the floating point representation of the integer",
                            msg.clone(),
                        ),
                        into_suggestion,
                        Applicability::MachineApplicable,
                    );
                } else if literal_is_ty_suffixed(expr) {
                    err.multipart_suggestion_verbose(
                        lit_msg,
                        suffix_suggestion,
                        Applicability::MachineApplicable,
                    );
                } else {
                    // Missing try_into implementation for `{integer}` to `{float}`
                    err.multipart_suggestion_verbose(
                        format!(
                            "{}, producing the floating point representation of the integer, \
                                rounded if necessary",
                            &msg,
                        ),
                        cast_suggestion,
                        Applicability::MaybeIncorrect, // lossy conversion
                    );
                }
                true
            }
            (
                &ty::Uint(ty::UintTy::U32 | ty::UintTy::U64 | ty::UintTy::U128)
                | &ty::Int(ty::IntTy::I32 | ty::IntTy::I64 | ty::IntTy::I128),
                &ty::Char,
            ) => {
                err.multipart_suggestion_verbose(
                    format!("{cast_msg}, since a `char` always occupies 4 bytes"),
                    cast_suggestion,
                    Applicability::MachineApplicable,
                );
                true
            }
            _ => false,
        }
    }

    /// Identify when the user has written `foo..bar()` instead of `foo.bar()`.
    pub(crate) fn suggest_method_call_on_range_literal(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'tcx>,
        checked_ty: Ty<'tcx>,
        expected_ty: Ty<'tcx>,
    ) {
        if !hir::is_range_literal(expr) {
            return;
        }
        let hir::ExprKind::Struct(hir::QPath::LangItem(LangItem::Range, ..), [start, end], _) =
            expr.kind
        else {
            return;
        };
        if let hir::Node::ExprField(_) = self.tcx.parent_hir_node(expr.hir_id) {
            // Ignore `Foo { field: a..Default::default() }`
            return;
        }
        let mut expr = end.expr;
        let mut expectation = Some(expected_ty);
        while let hir::ExprKind::MethodCall(_, rcvr, ..) = expr.kind {
            // Getting to the root receiver and asserting it is a fn call let's us ignore cases in
            // `tests/ui/methods/issues/issue-90315.stderr`.
            expr = rcvr;
            // If we have more than one layer of calls, then the expected ty
            // cannot guide the method probe.
            expectation = None;
        }
        let hir::ExprKind::Call(method_name, _) = expr.kind else {
            return;
        };
        let ty::Adt(adt, _) = checked_ty.kind() else {
            return;
        };
        if self.tcx.lang_items().range_struct() != Some(adt.did()) {
            return;
        }
        if let ty::Adt(adt, _) = expected_ty.kind()
            && self.tcx.is_lang_item(adt.did(), LangItem::Range)
        {
            return;
        }
        // Check if start has method named end.
        let hir::ExprKind::Path(hir::QPath::Resolved(None, p)) = method_name.kind else {
            return;
        };
        let [hir::PathSegment { ident, .. }] = p.segments else {
            return;
        };
        let self_ty = self.typeck_results.borrow().expr_ty(start.expr);
        let Ok(_pick) = self.lookup_probe_for_diagnostic(
            *ident,
            self_ty,
            expr,
            probe::ProbeScope::AllTraits,
            expectation,
        ) else {
            return;
        };
        let mut sugg = ".";
        let mut span = start.expr.span.between(end.expr.span);
        if span.lo() + BytePos(2) == span.hi() {
            // There's no space between the start, the range op and the end, suggest removal which
            // will be more noticeable than the replacement of `..` with `.`.
            span = span.with_lo(span.lo() + BytePos(1));
            sugg = "";
        }
        err.span_suggestion_verbose(
            span,
            "you likely meant to write a method call instead of a range",
            sugg,
            Applicability::MachineApplicable,
        );
    }

    /// Identify when the type error is because `()` is found in a binding that was assigned a
    /// block without a tail expression.
    pub(crate) fn suggest_return_binding_for_missing_tail_expr(
        &self,
        err: &mut Diag<'_>,
        expr: &hir::Expr<'_>,
        checked_ty: Ty<'tcx>,
        expected_ty: Ty<'tcx>,
    ) {
        if !checked_ty.is_unit() {
            return;
        }
        let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = expr.kind else {
            return;
        };
        let hir::def::Res::Local(hir_id) = path.res else {
            return;
        };
        let hir::Node::Pat(pat) = self.tcx.hir_node(hir_id) else {
            return;
        };
        let hir::Node::LetStmt(hir::LetStmt { ty: None, init: Some(init), .. }) =
            self.tcx.parent_hir_node(pat.hir_id)
        else {
            return;
        };
        let hir::ExprKind::Block(block, None) = init.kind else {
            return;
        };
        if block.expr.is_some() {
            return;
        }
        let [.., stmt] = block.stmts else {
            err.span_label(block.span, "this empty block is missing a tail expression");
            return;
        };
        let hir::StmtKind::Semi(tail_expr) = stmt.kind else {
            return;
        };
        let Some(ty) = self.node_ty_opt(tail_expr.hir_id) else {
            return;
        };
        if self.can_eq(self.param_env, expected_ty, ty)
            // FIXME: this happens with macro calls. Need to figure out why the stmt
            // `println!();` doesn't include the `;` in its `Span`. (#133845)
            // We filter these out to avoid ICEs with debug assertions on caused by
            // empty suggestions.
            && stmt.span.hi() != tail_expr.span.hi()
        {
            err.span_suggestion_short(
                stmt.span.with_lo(tail_expr.span.hi()),
                "remove this semicolon",
                "",
                Applicability::MachineApplicable,
            );
        } else {
            err.span_label(block.span, "this block is missing a tail expression");
        }
    }

    pub(crate) fn suggest_swapping_lhs_and_rhs(
        &self,
        err: &mut Diag<'_>,
        rhs_ty: Ty<'tcx>,
        lhs_ty: Ty<'tcx>,
        rhs_expr: &'tcx hir::Expr<'tcx>,
        lhs_expr: &'tcx hir::Expr<'tcx>,
    ) {
        if let Some(partial_eq_def_id) = self.infcx.tcx.lang_items().eq_trait()
            && self
                .infcx
                .type_implements_trait(partial_eq_def_id, [rhs_ty, lhs_ty], self.param_env)
                .must_apply_modulo_regions()
        {
            let sm = self.tcx.sess.source_map();
            if let Ok(rhs_snippet) = sm.span_to_snippet(rhs_expr.span)
                && let Ok(lhs_snippet) = sm.span_to_snippet(lhs_expr.span)
            {
                err.note(format!("`{rhs_ty}` implements `PartialEq<{lhs_ty}>`"));
                err.multipart_suggestion(
                    "consider swapping the equality",
                    vec![(lhs_expr.span, rhs_snippet), (rhs_expr.span, lhs_snippet)],
                    Applicability::MaybeIncorrect,
                );
            }
        }
    }
}
