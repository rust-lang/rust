use core::ops::ControlFlow;

use hir::def::CtorKind;
use hir::intravisit::{Visitor, walk_expr, walk_stmt};
use hir::{LetStmt, QPath};
use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::{Applicability, Diag};
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::{MatchSource, Node};
use rustc_middle::traits::{MatchExpressionArmCause, ObligationCause, ObligationCauseCode};
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self as ty, GenericArgKind, IsSuggestable, Ty, TypeVisitableExt};
use rustc_span::{Span, sym};
use tracing::debug;

use crate::error_reporting::TypeErrCtxt;
use crate::error_reporting::infer::hir::Path;
use crate::errors::{
    ConsiderAddingAwait, FnConsiderCasting, FnConsiderCastingBoth, FnItemsAreDistinct, FnUniqTypes,
    FunctionPointerSuggestion, SuggestAccessingField, SuggestRemoveSemiOrReturnBinding,
    SuggestTuplePatternMany, SuggestTuplePatternOne, TypeErrorAdditionalDiags,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum StatementAsExpression {
    CorrectType,
    NeedsBoxing,
}

#[derive(Clone, Copy)]
enum SuggestAsRefKind {
    Option,
    Result,
}

impl<'tcx> TypeErrCtxt<'_, 'tcx> {
    pub(super) fn suggest_remove_semi_or_return_binding(
        &self,
        first_id: Option<hir::HirId>,
        first_ty: Ty<'tcx>,
        first_span: Span,
        second_id: Option<hir::HirId>,
        second_ty: Ty<'tcx>,
        second_span: Span,
    ) -> Option<SuggestRemoveSemiOrReturnBinding> {
        let remove_semicolon = [
            (first_id, self.resolve_vars_if_possible(second_ty)),
            (second_id, self.resolve_vars_if_possible(first_ty)),
        ]
        .into_iter()
        .find_map(|(id, ty)| {
            let hir::Node::Block(blk) = self.tcx.hir_node(id?) else { return None };
            self.could_remove_semicolon(blk, ty)
        });
        match remove_semicolon {
            Some((sp, StatementAsExpression::NeedsBoxing)) => {
                Some(SuggestRemoveSemiOrReturnBinding::RemoveAndBox {
                    first_lo: first_span.shrink_to_lo(),
                    first_hi: first_span.shrink_to_hi(),
                    second_lo: second_span.shrink_to_lo(),
                    second_hi: second_span.shrink_to_hi(),
                    sp,
                })
            }
            Some((sp, StatementAsExpression::CorrectType)) => {
                Some(SuggestRemoveSemiOrReturnBinding::Remove { sp })
            }
            None => {
                let mut ret = None;
                for (id, ty) in [(first_id, second_ty), (second_id, first_ty)] {
                    if let Some(id) = id
                        && let hir::Node::Block(blk) = self.tcx.hir_node(id)
                        && let Some(diag) = self.consider_returning_binding_diag(blk, ty)
                    {
                        ret = Some(diag);
                        break;
                    }
                }
                ret
            }
        }
    }

    pub(super) fn suggest_tuple_pattern(
        &self,
        cause: &ObligationCause<'tcx>,
        exp_found: &ty::error::ExpectedFound<Ty<'tcx>>,
        diag: &mut Diag<'_>,
    ) {
        // Heavily inspired by `FnCtxt::suggest_compatible_variants`, with
        // some modifications due to that being in typeck and this being in infer.
        if let ObligationCauseCode::Pattern { .. } = cause.code() {
            if let ty::Adt(expected_adt, args) = exp_found.expected.kind() {
                let compatible_variants: Vec<_> = expected_adt
                    .variants()
                    .iter()
                    .filter(|variant| {
                        variant.fields.len() == 1 && variant.ctor_kind() == Some(CtorKind::Fn)
                    })
                    .filter_map(|variant| {
                        let sole_field = &variant.single_field();
                        let sole_field_ty = sole_field.ty(self.tcx, args);
                        if self.same_type_modulo_infer(sole_field_ty, exp_found.found) {
                            let variant_path =
                                with_no_trimmed_paths!(self.tcx.def_path_str(variant.def_id));
                            // FIXME #56861: DRYer prelude filtering
                            if let Some(path) = variant_path.strip_prefix("std::prelude::") {
                                if let Some((_, path)) = path.split_once("::") {
                                    return Some(path.to_string());
                                }
                            }
                            Some(variant_path)
                        } else {
                            None
                        }
                    })
                    .collect();
                match &compatible_variants[..] {
                    [] => {}
                    [variant] => {
                        let sugg = SuggestTuplePatternOne {
                            variant: variant.to_owned(),
                            span_low: cause.span.shrink_to_lo(),
                            span_high: cause.span.shrink_to_hi(),
                        };
                        diag.subdiagnostic(sugg);
                    }
                    _ => {
                        // More than one matching variant.
                        let sugg = SuggestTuplePatternMany {
                            path: self.tcx.def_path_str(expected_adt.did()),
                            cause_span: cause.span,
                            compatible_variants,
                        };
                        diag.subdiagnostic(sugg);
                    }
                }
            }
        }
    }

    /// A possible error is to forget to add `.await` when using futures:
    ///
    /// ```compile_fail,E0308
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
    pub(super) fn suggest_await_on_expect_found(
        &self,
        cause: &ObligationCause<'tcx>,
        exp_span: Span,
        exp_found: &ty::error::ExpectedFound<Ty<'tcx>>,
        diag: &mut Diag<'_>,
    ) {
        debug!(
            "suggest_await_on_expect_found: exp_span={:?}, expected_ty={:?}, found_ty={:?}",
            exp_span, exp_found.expected, exp_found.found,
        );

        match self.tcx.coroutine_kind(cause.body_id) {
            Some(hir::CoroutineKind::Desugared(
                hir::CoroutineDesugaring::Async | hir::CoroutineDesugaring::AsyncGen,
                _,
            )) => (),
            None
            | Some(
                hir::CoroutineKind::Coroutine(_)
                | hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Gen, _),
            ) => return,
        }

        if let ObligationCauseCode::CompareImplItem { .. } = cause.code() {
            return;
        }

        let subdiag = match (
            self.get_impl_future_output_ty(exp_found.expected),
            self.get_impl_future_output_ty(exp_found.found),
        ) {
            (Some(exp), Some(found)) if self.same_type_modulo_infer(exp, found) => match cause
                .code()
            {
                ObligationCauseCode::IfExpression { expr_id, .. } => {
                    let hir::Node::Expr(hir::Expr {
                        kind: hir::ExprKind::If(_, then_expr, _), ..
                    }) = self.tcx.hir_node(*expr_id)
                    else {
                        return;
                    };
                    let then_span = self.find_block_span_from_hir_id(then_expr.hir_id);
                    Some(ConsiderAddingAwait::BothFuturesSugg {
                        first: then_span.shrink_to_hi(),
                        second: exp_span.shrink_to_hi(),
                    })
                }
                ObligationCauseCode::MatchExpressionArm(box MatchExpressionArmCause {
                    prior_non_diverging_arms,
                    ..
                }) => {
                    if let [.., arm_span] = &prior_non_diverging_arms[..] {
                        Some(ConsiderAddingAwait::BothFuturesSugg {
                            first: arm_span.shrink_to_hi(),
                            second: exp_span.shrink_to_hi(),
                        })
                    } else {
                        Some(ConsiderAddingAwait::BothFuturesHelp)
                    }
                }
                _ => Some(ConsiderAddingAwait::BothFuturesHelp),
            },
            (_, Some(ty)) if self.same_type_modulo_infer(exp_found.expected, ty) => {
                // FIXME: Seems like we can't have a suggestion and a note with different spans in a single subdiagnostic
                diag.subdiagnostic(ConsiderAddingAwait::FutureSugg {
                    span: exp_span.shrink_to_hi(),
                });
                Some(ConsiderAddingAwait::FutureSuggNote { span: exp_span })
            }
            (Some(ty), _) if self.same_type_modulo_infer(ty, exp_found.found) => match cause.code()
            {
                ObligationCauseCode::Pattern { span: Some(then_span), origin_expr, .. } => {
                    origin_expr.is_some().then_some(ConsiderAddingAwait::FutureSugg {
                        span: then_span.shrink_to_hi(),
                    })
                }
                ObligationCauseCode::IfExpression { expr_id, .. } => {
                    let hir::Node::Expr(hir::Expr {
                        kind: hir::ExprKind::If(_, then_expr, _), ..
                    }) = self.tcx.hir_node(*expr_id)
                    else {
                        return;
                    };
                    let then_span = self.find_block_span_from_hir_id(then_expr.hir_id);
                    Some(ConsiderAddingAwait::FutureSugg { span: then_span.shrink_to_hi() })
                }
                ObligationCauseCode::MatchExpressionArm(box MatchExpressionArmCause {
                    prior_non_diverging_arms,
                    ..
                }) => Some({
                    ConsiderAddingAwait::FutureSuggMultiple {
                        spans: prior_non_diverging_arms
                            .iter()
                            .map(|arm| arm.shrink_to_hi())
                            .collect(),
                    }
                }),
                _ => None,
            },
            _ => None,
        };
        if let Some(subdiag) = subdiag {
            diag.subdiagnostic(subdiag);
        }
    }

    pub(super) fn suggest_accessing_field_where_appropriate(
        &self,
        cause: &ObligationCause<'tcx>,
        exp_found: &ty::error::ExpectedFound<Ty<'tcx>>,
        diag: &mut Diag<'_>,
    ) {
        debug!(
            "suggest_accessing_field_where_appropriate(cause={:?}, exp_found={:?})",
            cause, exp_found
        );
        if let ty::Adt(expected_def, expected_args) = exp_found.expected.kind() {
            if expected_def.is_enum() {
                return;
            }

            if let Some((name, ty)) = expected_def
                .non_enum_variant()
                .fields
                .iter()
                .filter(|field| field.vis.is_accessible_from(field.did, self.tcx))
                .map(|field| (field.name, field.ty(self.tcx, expected_args)))
                .find(|(_, ty)| self.same_type_modulo_infer(*ty, exp_found.found))
            {
                if let ObligationCauseCode::Pattern { span: Some(span), .. } = *cause.code() {
                    if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
                        let suggestion = if expected_def.is_struct() {
                            SuggestAccessingField::Safe { span, snippet, name, ty }
                        } else if expected_def.is_union() {
                            SuggestAccessingField::Unsafe { span, snippet, name, ty }
                        } else {
                            return;
                        };
                        diag.subdiagnostic(suggestion);
                    }
                }
            }
        }
    }

    pub(super) fn suggest_turning_stmt_into_expr(
        &self,
        cause: &ObligationCause<'tcx>,
        exp_found: &ty::error::ExpectedFound<Ty<'tcx>>,
        diag: &mut Diag<'_>,
    ) {
        let ty::error::ExpectedFound { expected, found } = exp_found;
        if !found.peel_refs().is_unit() {
            return;
        }

        let ObligationCauseCode::BlockTailExpression(hir_id, MatchSource::Normal) = cause.code()
        else {
            return;
        };

        let node = self.tcx.hir_node(*hir_id);
        let mut blocks = vec![];
        if let hir::Node::Block(block) = node
            && let Some(expr) = block.expr
            && let hir::ExprKind::Path(QPath::Resolved(_, Path { res, .. })) = expr.kind
            && let Res::Local(local) = res
            && let Node::LetStmt(LetStmt { init: Some(init), .. }) =
                self.tcx.parent_hir_node(*local)
        {
            fn collect_blocks<'hir>(expr: &hir::Expr<'hir>, blocks: &mut Vec<&hir::Block<'hir>>) {
                match expr.kind {
                    // `blk1` and `blk2` must be have the same types, it will be reported before reaching here
                    hir::ExprKind::If(_, blk1, Some(blk2)) => {
                        collect_blocks(blk1, blocks);
                        collect_blocks(blk2, blocks);
                    }
                    hir::ExprKind::Match(_, arms, _) => {
                        // all arms must have same types
                        for arm in arms.iter() {
                            collect_blocks(arm.body, blocks);
                        }
                    }
                    hir::ExprKind::Block(blk, _) => {
                        blocks.push(blk);
                    }
                    _ => {}
                }
            }
            collect_blocks(init, &mut blocks);
        }

        let expected_inner: Ty<'_> = expected.peel_refs();
        for block in blocks.iter() {
            self.consider_removing_semicolon(block, expected_inner, diag);
        }
    }

    /// A common error is to add an extra semicolon:
    ///
    /// ```compile_fail,E0308
    /// fn foo() -> usize {
    ///     22;
    /// }
    /// ```
    ///
    /// This routine checks if the final statement in a block is an
    /// expression with an explicit semicolon whose type is compatible
    /// with `expected_ty`. If so, it suggests removing the semicolon.
    pub fn consider_removing_semicolon(
        &self,
        blk: &'tcx hir::Block<'tcx>,
        expected_ty: Ty<'tcx>,
        diag: &mut Diag<'_>,
    ) -> bool {
        if let Some((span_semi, boxed)) = self.could_remove_semicolon(blk, expected_ty) {
            if let StatementAsExpression::NeedsBoxing = boxed {
                diag.span_suggestion_verbose(
                    span_semi,
                    "consider removing this semicolon and boxing the expression",
                    "",
                    Applicability::HasPlaceholders,
                );
            } else {
                diag.span_suggestion_short(
                    span_semi,
                    "remove this semicolon to return this value",
                    "",
                    Applicability::MachineApplicable,
                );
            }
            true
        } else {
            false
        }
    }

    pub(crate) fn suggest_function_pointers_impl(
        &self,
        span: Option<Span>,
        exp_found: &ty::error::ExpectedFound<Ty<'tcx>>,
        diag: &mut Diag<'_>,
    ) {
        let ty::error::ExpectedFound { expected, found } = exp_found;
        let expected_inner = expected.peel_refs();
        let found_inner = found.peel_refs();
        if !expected_inner.is_fn() || !found_inner.is_fn() {
            return;
        }
        match (expected_inner.kind(), found_inner.kind()) {
            (ty::FnPtr(sig_tys, hdr), ty::FnDef(did, args)) => {
                let sig = sig_tys.with(*hdr);
                let expected_sig = &(self.normalize_fn_sig)(sig);
                let found_sig =
                    &(self.normalize_fn_sig)(self.tcx.fn_sig(*did).instantiate(self.tcx, args));

                let fn_name = self.tcx.def_path_str_with_args(*did, args);

                if !self.same_type_modulo_infer(*found_sig, *expected_sig)
                    || !sig.is_suggestable(self.tcx, true)
                    || self.tcx.intrinsic(*did).is_some()
                {
                    return;
                }

                let Some(span) = span else {
                    let casting = format!("{fn_name} as {sig}");
                    diag.subdiagnostic(FnItemsAreDistinct);
                    diag.subdiagnostic(FnConsiderCasting { casting });
                    return;
                };

                let sugg = match (expected.is_ref(), found.is_ref()) {
                    (true, false) => {
                        FunctionPointerSuggestion::UseRef { span: span.shrink_to_lo() }
                    }
                    (false, true) => FunctionPointerSuggestion::RemoveRef { span, fn_name },
                    (true, true) => {
                        diag.subdiagnostic(FnItemsAreDistinct);
                        FunctionPointerSuggestion::CastRef { span, fn_name, sig }
                    }
                    (false, false) => {
                        diag.subdiagnostic(FnItemsAreDistinct);
                        FunctionPointerSuggestion::Cast { span: span.shrink_to_hi(), sig }
                    }
                };
                diag.subdiagnostic(sugg);
            }
            (ty::FnDef(did1, args1), ty::FnDef(did2, args2)) => {
                let expected_sig =
                    &(self.normalize_fn_sig)(self.tcx.fn_sig(*did1).instantiate(self.tcx, args1));
                let found_sig =
                    &(self.normalize_fn_sig)(self.tcx.fn_sig(*did2).instantiate(self.tcx, args2));

                if self.same_type_modulo_infer(*expected_sig, *found_sig) {
                    diag.subdiagnostic(FnUniqTypes);
                }

                if !self.same_type_modulo_infer(*found_sig, *expected_sig)
                    || !found_sig.is_suggestable(self.tcx, true)
                    || !expected_sig.is_suggestable(self.tcx, true)
                    || self.tcx.intrinsic(*did1).is_some()
                    || self.tcx.intrinsic(*did2).is_some()
                {
                    return;
                }

                let fn_name = self.tcx.def_path_str_with_args(*did2, args2);

                let Some(span) = span else {
                    diag.subdiagnostic(FnConsiderCastingBoth { sig: *expected_sig });
                    return;
                };

                let sug = if found.is_ref() {
                    FunctionPointerSuggestion::CastBothRef {
                        span,
                        fn_name,
                        found_sig: *found_sig,
                        expected_sig: *expected_sig,
                    }
                } else {
                    FunctionPointerSuggestion::CastBoth {
                        span: span.shrink_to_hi(),
                        found_sig: *found_sig,
                        expected_sig: *expected_sig,
                    }
                };

                diag.subdiagnostic(sug);
            }
            (ty::FnDef(did, args), ty::FnPtr(sig_tys, hdr)) => {
                let expected_sig =
                    &(self.normalize_fn_sig)(self.tcx.fn_sig(*did).instantiate(self.tcx, args));
                let found_sig = &(self.normalize_fn_sig)(sig_tys.with(*hdr));

                if !self.same_type_modulo_infer(*found_sig, *expected_sig) {
                    return;
                }

                let fn_name = self.tcx.def_path_str_with_args(*did, args);

                let casting = if expected.is_ref() {
                    format!("&({fn_name} as {found_sig})")
                } else {
                    format!("{fn_name} as {found_sig}")
                };

                diag.subdiagnostic(FnConsiderCasting { casting });
            }
            _ => {
                return;
            }
        };
    }

    pub(super) fn suggest_function_pointers(
        &self,
        cause: &ObligationCause<'tcx>,
        span: Span,
        exp_found: &ty::error::ExpectedFound<Ty<'tcx>>,
        terr: TypeError<'tcx>,
        diag: &mut Diag<'_>,
    ) {
        debug!("suggest_function_pointers(cause={:?}, exp_found={:?})", cause, exp_found);

        if exp_found.expected.peel_refs().is_fn() && exp_found.found.peel_refs().is_fn() {
            self.suggest_function_pointers_impl(Some(span), exp_found, diag);
        } else if let TypeError::Sorts(exp_found) = terr {
            self.suggest_function_pointers_impl(None, &exp_found, diag);
        }
    }

    fn should_suggest_as_ref_kind(
        &self,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) -> Option<SuggestAsRefKind> {
        if let (ty::Adt(exp_def, exp_args), ty::Ref(_, found_ty, _)) =
            (expected.kind(), found.kind())
        {
            if let ty::Adt(found_def, found_args) = *found_ty.kind() {
                if exp_def == &found_def {
                    let have_as_ref = &[
                        (sym::Option, SuggestAsRefKind::Option),
                        (sym::Result, SuggestAsRefKind::Result),
                    ];
                    if let Some(msg) = have_as_ref.iter().find_map(|(name, msg)| {
                        self.tcx.is_diagnostic_item(*name, exp_def.did()).then_some(msg)
                    }) {
                        let mut show_suggestion = true;
                        for (exp_ty, found_ty) in
                            std::iter::zip(exp_args.types(), found_args.types())
                        {
                            match *exp_ty.kind() {
                                ty::Ref(_, exp_ty, _) => {
                                    match (exp_ty.kind(), found_ty.kind()) {
                                        (_, ty::Param(_))
                                        | (_, ty::Infer(_))
                                        | (ty::Param(_), _)
                                        | (ty::Infer(_), _) => {}
                                        _ if self.same_type_modulo_infer(exp_ty, found_ty) => {}
                                        _ => show_suggestion = false,
                                    };
                                }
                                ty::Param(_) | ty::Infer(_) => {}
                                _ => show_suggestion = false,
                            }
                        }
                        if show_suggestion {
                            return Some(*msg);
                        }
                    }
                }
            }
        }
        None
    }

    // FIXME: Remove once `rustc_hir_typeck` is migrated to diagnostic structs
    pub fn should_suggest_as_ref(&self, expected: Ty<'tcx>, found: Ty<'tcx>) -> Option<&str> {
        match self.should_suggest_as_ref_kind(expected, found) {
            Some(SuggestAsRefKind::Option) => Some(
                "you can convert from `&Option<T>` to `Option<&T>` using \
            `.as_ref()`",
            ),
            Some(SuggestAsRefKind::Result) => Some(
                "you can convert from `&Result<T, E>` to \
            `Result<&T, &E>` using `.as_ref()`",
            ),
            None => None,
        }
    }
    /// Try to find code with pattern `if Some(..) = expr`
    /// use a `visitor` to mark the `if` which its span contains given error span,
    /// and then try to find a assignment in the `cond` part, which span is equal with error span
    pub(super) fn suggest_let_for_letchains(
        &self,
        cause: &ObligationCause<'_>,
        span: Span,
    ) -> Option<TypeErrorAdditionalDiags> {
        /// Find the if expression with given span
        struct IfVisitor {
            found_if: bool,
            err_span: Span,
        }

        impl<'v> Visitor<'v> for IfVisitor {
            type Result = ControlFlow<()>;
            fn visit_expr(&mut self, ex: &'v hir::Expr<'v>) -> Self::Result {
                match ex.kind {
                    hir::ExprKind::If(cond, _, _) => {
                        self.found_if = true;
                        walk_expr(self, cond)?;
                        self.found_if = false;
                        ControlFlow::Continue(())
                    }
                    _ => walk_expr(self, ex),
                }
            }

            fn visit_stmt(&mut self, ex: &'v hir::Stmt<'v>) -> Self::Result {
                if let hir::StmtKind::Let(LetStmt {
                    span,
                    pat: hir::Pat { .. },
                    ty: None,
                    init: Some(_),
                    ..
                }) = &ex.kind
                    && self.found_if
                    && span.eq(&self.err_span)
                {
                    ControlFlow::Break(())
                } else {
                    walk_stmt(self, ex)
                }
            }
        }

        self.tcx.hir_maybe_body_owned_by(cause.body_id).and_then(|body| {
            IfVisitor { err_span: span, found_if: false }
                .visit_body(&body)
                .is_break()
                .then(|| TypeErrorAdditionalDiags::AddLetForLetChains { span: span.shrink_to_lo() })
        })
    }

    /// For "one type is more general than the other" errors on closures, suggest changing the lifetime
    /// of the parameters to accept all lifetimes.
    pub(super) fn suggest_for_all_lifetime_closure(
        &self,
        span: Span,
        hir: hir::Node<'_>,
        exp_found: &ty::error::ExpectedFound<ty::TraitRef<'tcx>>,
        diag: &mut Diag<'_>,
    ) {
        // 0. Extract fn_decl from hir
        let hir::Node::Expr(hir::Expr {
            kind: hir::ExprKind::Closure(hir::Closure { body, fn_decl, .. }),
            ..
        }) = hir
        else {
            return;
        };
        let hir::Body { params, .. } = self.tcx.hir_body(*body);

        // 1. Get the args of the closure.
        // 2. Assume exp_found is FnOnce / FnMut / Fn, we can extract function parameters from [1].
        let Some(expected) = exp_found.expected.args.get(1) else {
            return;
        };
        let Some(found) = exp_found.found.args.get(1) else {
            return;
        };
        let expected = expected.kind();
        let found = found.kind();
        // 3. Extract the tuple type from Fn trait and suggest the change.
        if let GenericArgKind::Type(expected) = expected
            && let GenericArgKind::Type(found) = found
            && let ty::Tuple(expected) = expected.kind()
            && let ty::Tuple(found) = found.kind()
            && expected.len() == found.len()
        {
            let mut suggestion = "|".to_string();
            let mut is_first = true;
            let mut has_suggestion = false;

            for (((expected, found), param_hir), arg_hir) in
                expected.iter().zip(found.iter()).zip(params.iter()).zip(fn_decl.inputs.iter())
            {
                if is_first {
                    is_first = false;
                } else {
                    suggestion += ", ";
                }

                if let ty::Ref(expected_region, _, _) = expected.kind()
                    && let ty::Ref(found_region, _, _) = found.kind()
                    && expected_region.is_bound()
                    && !found_region.is_bound()
                    && let hir::TyKind::Infer(()) = arg_hir.kind
                {
                    // If the expected region is late bound, the found region is not, and users are asking compiler
                    // to infer the type, we can suggest adding `: &_`.
                    if param_hir.pat.span == param_hir.ty_span {
                        // for `|x|`, `|_|`, `|x: impl Foo|`
                        let Ok(pat) =
                            self.tcx.sess.source_map().span_to_snippet(param_hir.pat.span)
                        else {
                            return;
                        };
                        suggestion += &format!("{pat}: &_");
                    } else {
                        // for `|x: ty|`, `|_: ty|`
                        let Ok(pat) =
                            self.tcx.sess.source_map().span_to_snippet(param_hir.pat.span)
                        else {
                            return;
                        };
                        let Ok(ty) = self.tcx.sess.source_map().span_to_snippet(param_hir.ty_span)
                        else {
                            return;
                        };
                        suggestion += &format!("{pat}: &{ty}");
                    }
                    has_suggestion = true;
                } else {
                    let Ok(arg) = self.tcx.sess.source_map().span_to_snippet(param_hir.span) else {
                        return;
                    };
                    // Otherwise, keep it as-is.
                    suggestion += &arg;
                }
            }
            suggestion += "|";

            if has_suggestion {
                diag.span_suggestion_verbose(
                    span,
                    "consider specifying the type of the closure parameters",
                    suggestion,
                    Applicability::MaybeIncorrect,
                );
            }
        }
    }
}

impl<'tcx> TypeErrCtxt<'_, 'tcx> {
    /// Be helpful when the user wrote `{... expr; }` and taking the `;` off
    /// is enough to fix the error.
    fn could_remove_semicolon(
        &self,
        blk: &'tcx hir::Block<'tcx>,
        expected_ty: Ty<'tcx>,
    ) -> Option<(Span, StatementAsExpression)> {
        let blk = blk.innermost_block();
        // Do not suggest if we have a tail expr.
        if blk.expr.is_some() {
            return None;
        }
        let last_stmt = blk.stmts.last()?;
        let hir::StmtKind::Semi(last_expr) = last_stmt.kind else {
            return None;
        };
        let last_expr_ty = self.typeck_results.as_ref()?.expr_ty_opt(last_expr)?;
        let needs_box = match (last_expr_ty.kind(), expected_ty.kind()) {
            _ if last_expr_ty.references_error() => return None,
            _ if self.same_type_modulo_infer(last_expr_ty, expected_ty) => {
                StatementAsExpression::CorrectType
            }
            (
                ty::Alias(ty::Opaque, ty::AliasTy { def_id: last_def_id, .. }),
                ty::Alias(ty::Opaque, ty::AliasTy { def_id: exp_def_id, .. }),
            ) if last_def_id == exp_def_id => StatementAsExpression::CorrectType,
            (
                ty::Alias(ty::Opaque, ty::AliasTy { def_id: last_def_id, args: last_bounds, .. }),
                ty::Alias(ty::Opaque, ty::AliasTy { def_id: exp_def_id, args: exp_bounds, .. }),
            ) => {
                debug!(
                    "both opaque, likely future {:?} {:?} {:?} {:?}",
                    last_def_id, last_bounds, exp_def_id, exp_bounds
                );

                let last_local_id = last_def_id.as_local()?;
                let exp_local_id = exp_def_id.as_local()?;

                match (
                    &self.tcx.hir_expect_opaque_ty(last_local_id),
                    &self.tcx.hir_expect_opaque_ty(exp_local_id),
                ) {
                    (
                        hir::OpaqueTy { bounds: last_bounds, .. },
                        hir::OpaqueTy { bounds: exp_bounds, .. },
                    ) if std::iter::zip(*last_bounds, *exp_bounds).all(|(left, right)| match (
                        left, right,
                    ) {
                        // FIXME: Suspicious
                        (hir::GenericBound::Trait(tl), hir::GenericBound::Trait(tr))
                            if tl.trait_ref.trait_def_id() == tr.trait_ref.trait_def_id()
                                && tl.modifiers == tr.modifiers =>
                        {
                            true
                        }
                        _ => false,
                    }) =>
                    {
                        StatementAsExpression::NeedsBoxing
                    }
                    _ => StatementAsExpression::CorrectType,
                }
            }
            _ => return None,
        };
        let span = if last_stmt.span.from_expansion() {
            let mac_call = rustc_span::source_map::original_sp(last_stmt.span, blk.span);
            self.tcx.sess.source_map().mac_call_stmt_semi_span(mac_call)?
        } else {
            self.tcx
                .sess
                .source_map()
                .span_extend_while_whitespace(last_expr.span)
                .shrink_to_hi()
                .with_hi(last_stmt.span.hi())
        };

        Some((span, needs_box))
    }

    /// Suggest returning a local binding with a compatible type if the block
    /// has no return expression.
    fn consider_returning_binding_diag(
        &self,
        blk: &'tcx hir::Block<'tcx>,
        expected_ty: Ty<'tcx>,
    ) -> Option<SuggestRemoveSemiOrReturnBinding> {
        let blk = blk.innermost_block();
        // Do not suggest if we have a tail expr.
        if blk.expr.is_some() {
            return None;
        }
        let mut shadowed = FxIndexSet::default();
        let mut candidate_idents = vec![];
        let mut find_compatible_candidates = |pat: &hir::Pat<'_>| {
            if let hir::PatKind::Binding(_, hir_id, ident, _) = &pat.kind
                && let Some(pat_ty) = self
                    .typeck_results
                    .as_ref()
                    .and_then(|typeck_results| typeck_results.node_type_opt(*hir_id))
            {
                let pat_ty = self.resolve_vars_if_possible(pat_ty);
                if self.same_type_modulo_infer(pat_ty, expected_ty)
                    && !(pat_ty, expected_ty).references_error()
                    && shadowed.insert(ident.name)
                {
                    candidate_idents.push((*ident, pat_ty));
                }
            }
            true
        };

        for stmt in blk.stmts.iter().rev() {
            let hir::StmtKind::Let(local) = &stmt.kind else {
                continue;
            };
            local.pat.walk(&mut find_compatible_candidates);
        }
        match self.tcx.parent_hir_node(blk.hir_id) {
            hir::Node::Expr(hir::Expr { hir_id, .. }) => match self.tcx.parent_hir_node(*hir_id) {
                hir::Node::Arm(hir::Arm { pat, .. }) => {
                    pat.walk(&mut find_compatible_candidates);
                }

                hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn { body, .. }, .. })
                | hir::Node::ImplItem(hir::ImplItem {
                    kind: hir::ImplItemKind::Fn(_, body), ..
                })
                | hir::Node::TraitItem(hir::TraitItem {
                    kind: hir::TraitItemKind::Fn(_, hir::TraitFn::Provided(body)),
                    ..
                })
                | hir::Node::Expr(hir::Expr {
                    kind: hir::ExprKind::Closure(hir::Closure { body, .. }),
                    ..
                }) => {
                    for param in self.tcx.hir_body(*body).params {
                        param.pat.walk(&mut find_compatible_candidates);
                    }
                }
                hir::Node::Expr(hir::Expr {
                    kind:
                        hir::ExprKind::If(
                            hir::Expr { kind: hir::ExprKind::Let(let_), .. },
                            then_block,
                            _,
                        ),
                    ..
                }) if then_block.hir_id == *hir_id => {
                    let_.pat.walk(&mut find_compatible_candidates);
                }
                _ => {}
            },
            _ => {}
        }

        match &candidate_idents[..] {
            [(ident, _ty)] => {
                let sm = self.tcx.sess.source_map();
                let (span, sugg) = if let Some(stmt) = blk.stmts.last() {
                    let stmt_span = sm.stmt_span(stmt.span, blk.span);
                    let sugg = if sm.is_multiline(blk.span)
                        && let Some(spacing) = sm.indentation_before(stmt_span)
                    {
                        format!("\n{spacing}{ident}")
                    } else {
                        format!(" {ident}")
                    };
                    (stmt_span.shrink_to_hi(), sugg)
                } else {
                    let sugg = if sm.is_multiline(blk.span)
                        && let Some(spacing) = sm.indentation_before(blk.span.shrink_to_lo())
                    {
                        format!("\n{spacing}    {ident}\n{spacing}")
                    } else {
                        format!(" {ident} ")
                    };
                    let left_span = sm.span_through_char(blk.span, '{').shrink_to_hi();
                    (sm.span_extend_while_whitespace(left_span), sugg)
                };
                Some(SuggestRemoveSemiOrReturnBinding::Add { sp: span, code: sugg, ident: *ident })
            }
            values if (1..3).contains(&values.len()) => {
                let spans = values.iter().map(|(ident, _)| ident.span).collect::<Vec<_>>();
                Some(SuggestRemoveSemiOrReturnBinding::AddOne { spans: spans.into() })
            }
            _ => None,
        }
    }

    pub fn consider_returning_binding(
        &self,
        blk: &'tcx hir::Block<'tcx>,
        expected_ty: Ty<'tcx>,
        err: &mut Diag<'_>,
    ) -> bool {
        let diag = self.consider_returning_binding_diag(blk, expected_ty);
        match diag {
            Some(diag) => {
                err.subdiagnostic(diag);
                true
            }
            None => false,
        }
    }
}
