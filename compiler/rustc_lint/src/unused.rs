use std::iter;

use rustc_ast::util::{classify, parser};
use rustc_ast::{self as ast, ExprKind, HasAttrs as _, StmtKind};
use rustc_attr_data_structures::{AttributeKind, find_attr};
use rustc_errors::{MultiSpan, pluralize};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::{self as hir, LangItem};
use rustc_infer::traits::util::elaborate;
use rustc_middle::ty::{self, Ty, adjustment};
use rustc_session::{declare_lint, declare_lint_pass, impl_lint_pass};
use rustc_span::{BytePos, Span, Symbol, kw, sym};
use tracing::instrument;

use crate::lints::{
    PathStatementDrop, PathStatementDropSub, PathStatementNoEffect, UnusedAllocationDiag,
    UnusedAllocationMutDiag, UnusedClosure, UnusedCoroutine, UnusedDef, UnusedDefSuggestion,
    UnusedDelim, UnusedDelimSuggestion, UnusedImportBracesDiag, UnusedOp, UnusedOpSuggestion,
    UnusedResult,
};
use crate::{EarlyContext, EarlyLintPass, LateContext, LateLintPass, Lint, LintContext};

declare_lint! {
    /// The `unused_must_use` lint detects unused result of a type flagged as
    /// `#[must_use]`.
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn returns_result() -> Result<(), ()> {
    ///     Ok(())
    /// }
    ///
    /// fn main() {
    ///     returns_result();
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The `#[must_use]` attribute is an indicator that it is a mistake to
    /// ignore the value. See [the reference] for more details.
    ///
    /// [the reference]: https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-must_use-attribute
    pub UNUSED_MUST_USE,
    Warn,
    "unused result of a type flagged as `#[must_use]`",
    report_in_external_macro
}

declare_lint! {
    /// The `unused_results` lint checks for the unused result of an
    /// expression in a statement.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(unused_results)]
    /// fn foo<T>() -> T { panic!() }
    ///
    /// fn main() {
    ///     foo::<usize>();
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Ignoring the return value of a function may indicate a mistake. In
    /// cases were it is almost certain that the result should be used, it is
    /// recommended to annotate the function with the [`must_use` attribute].
    /// Failure to use such a return value will trigger the [`unused_must_use`
    /// lint] which is warn-by-default. The `unused_results` lint is
    /// essentially the same, but triggers for *all* return values.
    ///
    /// This lint is "allow" by default because it can be noisy, and may not be
    /// an actual problem. For example, calling the `remove` method of a `Vec`
    /// or `HashMap` returns the previous value, which you may not care about.
    /// Using this lint would require explicitly ignoring or discarding such
    /// values.
    ///
    /// [`must_use` attribute]: https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-must_use-attribute
    /// [`unused_must_use` lint]: warn-by-default.html#unused-must-use
    pub UNUSED_RESULTS,
    Allow,
    "unused result of an expression in a statement"
}

declare_lint_pass!(UnusedResults => [UNUSED_MUST_USE, UNUSED_RESULTS]);

impl<'tcx> LateLintPass<'tcx> for UnusedResults {
    fn check_stmt(&mut self, cx: &LateContext<'_>, s: &hir::Stmt<'_>) {
        let hir::StmtKind::Semi(mut expr) = s.kind else {
            return;
        };

        let mut expr_is_from_block = false;
        while let hir::ExprKind::Block(blk, ..) = expr.kind
            && let hir::Block { expr: Some(e), .. } = blk
        {
            expr = e;
            expr_is_from_block = true;
        }

        if let hir::ExprKind::Ret(..) = expr.kind {
            return;
        }

        if let hir::ExprKind::Match(await_expr, _arms, hir::MatchSource::AwaitDesugar) = expr.kind
            && let ty = cx.typeck_results().expr_ty(await_expr)
            && let ty::Alias(ty::Opaque, ty::AliasTy { def_id: future_def_id, .. }) = ty.kind()
            && cx.tcx.ty_is_opaque_future(ty)
            && let async_fn_def_id = cx.tcx.parent(*future_def_id)
            && matches!(cx.tcx.def_kind(async_fn_def_id), DefKind::Fn | DefKind::AssocFn)
            // Check that this `impl Future` actually comes from an `async fn`
            && cx.tcx.asyncness(async_fn_def_id).is_async()
            && check_must_use_def(
                cx,
                async_fn_def_id,
                expr.span,
                "output of future returned by ",
                "",
                expr_is_from_block,
            )
        {
            // We have a bare `foo().await;` on an opaque type from an async function that was
            // annotated with `#[must_use]`.
            return;
        }

        let ty = cx.typeck_results().expr_ty(expr);

        let must_use_result = is_ty_must_use(cx, ty, expr, expr.span);
        let type_lint_emitted_or_suppressed = match must_use_result {
            Some(path) => {
                emit_must_use_untranslated(cx, &path, "", "", 1, false, expr_is_from_block);
                true
            }
            None => false,
        };

        let fn_warned = check_fn_must_use(cx, expr, expr_is_from_block);

        if !fn_warned && type_lint_emitted_or_suppressed {
            // We don't warn about unused unit or uninhabited types.
            // (See https://github.com/rust-lang/rust/issues/43806 for details.)
            return;
        }

        let must_use_op = match expr.kind {
            // Hardcoding operators here seemed more expedient than the
            // refactoring that would be needed to look up the `#[must_use]`
            // attribute which does exist on the comparison trait methods
            hir::ExprKind::Binary(bin_op, ..) => match bin_op.node {
                hir::BinOpKind::Eq
                | hir::BinOpKind::Lt
                | hir::BinOpKind::Le
                | hir::BinOpKind::Ne
                | hir::BinOpKind::Ge
                | hir::BinOpKind::Gt => Some("comparison"),
                hir::BinOpKind::Add
                | hir::BinOpKind::Sub
                | hir::BinOpKind::Div
                | hir::BinOpKind::Mul
                | hir::BinOpKind::Rem => Some("arithmetic operation"),
                hir::BinOpKind::And | hir::BinOpKind::Or => Some("logical operation"),
                hir::BinOpKind::BitXor
                | hir::BinOpKind::BitAnd
                | hir::BinOpKind::BitOr
                | hir::BinOpKind::Shl
                | hir::BinOpKind::Shr => Some("bitwise operation"),
            },
            hir::ExprKind::AddrOf(..) => Some("borrow"),
            hir::ExprKind::OffsetOf(..) => Some("`offset_of` call"),
            hir::ExprKind::Unary(..) => Some("unary operation"),
            _ => None,
        };

        let mut op_warned = false;

        if let Some(must_use_op) = must_use_op {
            cx.emit_span_lint(
                UNUSED_MUST_USE,
                expr.span,
                UnusedOp {
                    op: must_use_op,
                    label: expr.span,
                    suggestion: if expr_is_from_block {
                        UnusedOpSuggestion::BlockTailExpr {
                            before_span: expr.span.shrink_to_lo(),
                            after_span: expr.span.shrink_to_hi(),
                        }
                    } else {
                        UnusedOpSuggestion::NormalExpr { span: expr.span.shrink_to_lo() }
                    },
                },
            );
            op_warned = true;
        }

        if !(type_lint_emitted_or_suppressed || fn_warned || op_warned) {
            cx.emit_span_lint(UNUSED_RESULTS, s.span, UnusedResult { ty });
        }

        fn check_fn_must_use(
            cx: &LateContext<'_>,
            expr: &hir::Expr<'_>,
            expr_is_from_block: bool,
        ) -> bool {
            let maybe_def_id = match expr.kind {
                hir::ExprKind::Call(callee, _) => {
                    match callee.kind {
                        hir::ExprKind::Path(ref qpath) => {
                            match cx.qpath_res(qpath, callee.hir_id) {
                                Res::Def(DefKind::Fn | DefKind::AssocFn, def_id) => Some(def_id),
                                // `Res::Local` if it was a closure, for which we
                                // do not currently support must-use linting
                                _ => None,
                            }
                        }
                        _ => None,
                    }
                }
                hir::ExprKind::MethodCall(..) => {
                    cx.typeck_results().type_dependent_def_id(expr.hir_id)
                }
                _ => None,
            };
            if let Some(def_id) = maybe_def_id {
                check_must_use_def(
                    cx,
                    def_id,
                    expr.span,
                    "return value of ",
                    "",
                    expr_is_from_block,
                )
            } else {
                false
            }
        }

        /// A path through a type to a must_use source. Contains useful info for the lint.
        #[derive(Debug)]
        enum MustUsePath {
            /// Suppress must_use checking.
            Suppressed,
            /// The root of the normal must_use lint with an optional message.
            Def(Span, DefId, Option<Symbol>),
            Boxed(Box<Self>),
            Pinned(Box<Self>),
            Opaque(Box<Self>),
            TraitObject(Box<Self>),
            TupleElement(Vec<(usize, Self)>),
            Array(Box<Self>, u64),
            /// The root of the unused_closures lint.
            Closure(Span),
            /// The root of the unused_coroutines lint.
            Coroutine(Span),
        }

        #[instrument(skip(cx, expr), level = "debug", ret)]
        fn is_ty_must_use<'tcx>(
            cx: &LateContext<'tcx>,
            ty: Ty<'tcx>,
            expr: &hir::Expr<'_>,
            span: Span,
        ) -> Option<MustUsePath> {
            if ty.is_unit()
                || !ty.is_inhabited_from(
                    cx.tcx,
                    cx.tcx.parent_module(expr.hir_id).to_def_id(),
                    cx.typing_env(),
                )
            {
                return Some(MustUsePath::Suppressed);
            }

            match *ty.kind() {
                ty::Adt(..) if let Some(boxed) = ty.boxed_ty() => {
                    is_ty_must_use(cx, boxed, expr, span)
                        .map(|inner| MustUsePath::Boxed(Box::new(inner)))
                }
                ty::Adt(def, args) if cx.tcx.is_lang_item(def.did(), LangItem::Pin) => {
                    let pinned_ty = args.type_at(0);
                    is_ty_must_use(cx, pinned_ty, expr, span)
                        .map(|inner| MustUsePath::Pinned(Box::new(inner)))
                }
                ty::Adt(def, _) => is_def_must_use(cx, def.did(), span),
                ty::Alias(ty::Opaque | ty::Projection, ty::AliasTy { def_id: def, .. }) => {
                    elaborate(cx.tcx, cx.tcx.explicit_item_self_bounds(def).iter_identity_copied())
                        // We only care about self bounds for the impl-trait
                        .filter_only_self()
                        .find_map(|(pred, _span)| {
                            // We only look at the `DefId`, so it is safe to skip the binder here.
                            if let ty::ClauseKind::Trait(ref poly_trait_predicate) =
                                pred.kind().skip_binder()
                            {
                                let def_id = poly_trait_predicate.trait_ref.def_id;

                                is_def_must_use(cx, def_id, span)
                            } else {
                                None
                            }
                        })
                        .map(|inner| MustUsePath::Opaque(Box::new(inner)))
                }
                ty::Dynamic(binders, _, _) => binders.iter().find_map(|predicate| {
                    if let ty::ExistentialPredicate::Trait(ref trait_ref) = predicate.skip_binder()
                    {
                        let def_id = trait_ref.def_id;
                        is_def_must_use(cx, def_id, span)
                            .map(|inner| MustUsePath::TraitObject(Box::new(inner)))
                    } else {
                        None
                    }
                }),
                ty::Tuple(tys) => {
                    let elem_exprs = if let hir::ExprKind::Tup(elem_exprs) = expr.kind {
                        debug_assert_eq!(elem_exprs.len(), tys.len());
                        elem_exprs
                    } else {
                        &[]
                    };

                    // Default to `expr`.
                    let elem_exprs = elem_exprs.iter().chain(iter::repeat(expr));

                    let nested_must_use = tys
                        .iter()
                        .zip(elem_exprs)
                        .enumerate()
                        .filter_map(|(i, (ty, expr))| {
                            is_ty_must_use(cx, ty, expr, expr.span).map(|path| (i, path))
                        })
                        .collect::<Vec<_>>();

                    if !nested_must_use.is_empty() {
                        Some(MustUsePath::TupleElement(nested_must_use))
                    } else {
                        None
                    }
                }
                ty::Array(ty, len) => match len.try_to_target_usize(cx.tcx) {
                    // If the array is empty we don't lint, to avoid false positives
                    Some(0) | None => None,
                    // If the array is definitely non-empty, we can do `#[must_use]` checking.
                    Some(len) => is_ty_must_use(cx, ty, expr, span)
                        .map(|inner| MustUsePath::Array(Box::new(inner), len)),
                },
                ty::Closure(..) | ty::CoroutineClosure(..) => Some(MustUsePath::Closure(span)),
                ty::Coroutine(def_id, ..) => {
                    // async fn should be treated as "implementor of `Future`"
                    let must_use = if cx.tcx.coroutine_is_async(def_id) {
                        let def_id = cx.tcx.lang_items().future_trait()?;
                        is_def_must_use(cx, def_id, span)
                            .map(|inner| MustUsePath::Opaque(Box::new(inner)))
                    } else {
                        None
                    };
                    must_use.or(Some(MustUsePath::Coroutine(span)))
                }
                _ => None,
            }
        }

        fn is_def_must_use(cx: &LateContext<'_>, def_id: DefId, span: Span) -> Option<MustUsePath> {
            if let Some(reason) = find_attr!(
                cx.tcx.get_all_attrs(def_id),
                AttributeKind::MustUse { reason, .. } => reason
            ) {
                // check for #[must_use = "..."]
                Some(MustUsePath::Def(span, def_id, *reason))
            } else {
                None
            }
        }

        // Returns whether further errors should be suppressed because either a lint has been
        // emitted or the type should be ignored.
        fn check_must_use_def(
            cx: &LateContext<'_>,
            def_id: DefId,
            span: Span,
            descr_pre_path: &str,
            descr_post_path: &str,
            expr_is_from_block: bool,
        ) -> bool {
            is_def_must_use(cx, def_id, span)
                .map(|must_use_path| {
                    emit_must_use_untranslated(
                        cx,
                        &must_use_path,
                        descr_pre_path,
                        descr_post_path,
                        1,
                        false,
                        expr_is_from_block,
                    )
                })
                .is_some()
        }

        #[instrument(skip(cx), level = "debug")]
        fn emit_must_use_untranslated(
            cx: &LateContext<'_>,
            path: &MustUsePath,
            descr_pre: &str,
            descr_post: &str,
            plural_len: usize,
            is_inner: bool,
            expr_is_from_block: bool,
        ) {
            let plural_suffix = pluralize!(plural_len);

            match path {
                MustUsePath::Suppressed => {}
                MustUsePath::Boxed(path) => {
                    let descr_pre = &format!("{descr_pre}boxed ");
                    emit_must_use_untranslated(
                        cx,
                        path,
                        descr_pre,
                        descr_post,
                        plural_len,
                        true,
                        expr_is_from_block,
                    );
                }
                MustUsePath::Pinned(path) => {
                    let descr_pre = &format!("{descr_pre}pinned ");
                    emit_must_use_untranslated(
                        cx,
                        path,
                        descr_pre,
                        descr_post,
                        plural_len,
                        true,
                        expr_is_from_block,
                    );
                }
                MustUsePath::Opaque(path) => {
                    let descr_pre = &format!("{descr_pre}implementer{plural_suffix} of ");
                    emit_must_use_untranslated(
                        cx,
                        path,
                        descr_pre,
                        descr_post,
                        plural_len,
                        true,
                        expr_is_from_block,
                    );
                }
                MustUsePath::TraitObject(path) => {
                    let descr_post = &format!(" trait object{plural_suffix}{descr_post}");
                    emit_must_use_untranslated(
                        cx,
                        path,
                        descr_pre,
                        descr_post,
                        plural_len,
                        true,
                        expr_is_from_block,
                    );
                }
                MustUsePath::TupleElement(elems) => {
                    for (index, path) in elems {
                        let descr_post = &format!(" in tuple element {index}");
                        emit_must_use_untranslated(
                            cx,
                            path,
                            descr_pre,
                            descr_post,
                            plural_len,
                            true,
                            expr_is_from_block,
                        );
                    }
                }
                MustUsePath::Array(path, len) => {
                    let descr_pre = &format!("{descr_pre}array{plural_suffix} of ");
                    emit_must_use_untranslated(
                        cx,
                        path,
                        descr_pre,
                        descr_post,
                        plural_len.saturating_add(usize::try_from(*len).unwrap_or(usize::MAX)),
                        true,
                        expr_is_from_block,
                    );
                }
                MustUsePath::Closure(span) => {
                    cx.emit_span_lint(
                        UNUSED_MUST_USE,
                        *span,
                        UnusedClosure { count: plural_len, pre: descr_pre, post: descr_post },
                    );
                }
                MustUsePath::Coroutine(span) => {
                    cx.emit_span_lint(
                        UNUSED_MUST_USE,
                        *span,
                        UnusedCoroutine { count: plural_len, pre: descr_pre, post: descr_post },
                    );
                }
                MustUsePath::Def(span, def_id, reason) => {
                    cx.emit_span_lint(
                        UNUSED_MUST_USE,
                        *span,
                        UnusedDef {
                            pre: descr_pre,
                            post: descr_post,
                            cx,
                            def_id: *def_id,
                            note: *reason,
                            suggestion: (!is_inner).then_some(if expr_is_from_block {
                                UnusedDefSuggestion::BlockTailExpr {
                                    before_span: span.shrink_to_lo(),
                                    after_span: span.shrink_to_hi(),
                                }
                            } else {
                                UnusedDefSuggestion::NormalExpr { span: span.shrink_to_lo() }
                            }),
                        },
                    );
                }
            }
        }
    }
}

declare_lint! {
    /// The `path_statements` lint detects path statements with no effect.
    ///
    /// ### Example
    ///
    /// ```rust
    /// let x = 42;
    ///
    /// x;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// It is usually a mistake to have a statement that has no effect.
    pub PATH_STATEMENTS,
    Warn,
    "path statements with no effect"
}

declare_lint_pass!(PathStatements => [PATH_STATEMENTS]);

impl<'tcx> LateLintPass<'tcx> for PathStatements {
    fn check_stmt(&mut self, cx: &LateContext<'_>, s: &hir::Stmt<'_>) {
        if let hir::StmtKind::Semi(expr) = s.kind {
            if let hir::ExprKind::Path(_) = expr.kind {
                let ty = cx.typeck_results().expr_ty(expr);
                if ty.needs_drop(cx.tcx, cx.typing_env()) {
                    let sub = if let Ok(snippet) = cx.sess().source_map().span_to_snippet(expr.span)
                    {
                        PathStatementDropSub::Suggestion { span: s.span, snippet }
                    } else {
                        PathStatementDropSub::Help { span: s.span }
                    };
                    cx.emit_span_lint(PATH_STATEMENTS, s.span, PathStatementDrop { sub })
                } else {
                    cx.emit_span_lint(PATH_STATEMENTS, s.span, PathStatementNoEffect);
                }
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum UnusedDelimsCtx {
    FunctionArg,
    MethodArg,
    AssignedValue,
    AssignedValueLetElse,
    IfCond,
    WhileCond,
    ForIterExpr,
    MatchScrutineeExpr,
    ReturnValue,
    BlockRetValue,
    BreakValue,
    LetScrutineeExpr,
    ArrayLenExpr,
    AnonConst,
    MatchArmExpr,
    IndexExpr,
}

impl From<UnusedDelimsCtx> for &'static str {
    fn from(ctx: UnusedDelimsCtx) -> &'static str {
        match ctx {
            UnusedDelimsCtx::FunctionArg => "function argument",
            UnusedDelimsCtx::MethodArg => "method argument",
            UnusedDelimsCtx::AssignedValue | UnusedDelimsCtx::AssignedValueLetElse => {
                "assigned value"
            }
            UnusedDelimsCtx::IfCond => "`if` condition",
            UnusedDelimsCtx::WhileCond => "`while` condition",
            UnusedDelimsCtx::ForIterExpr => "`for` iterator expression",
            UnusedDelimsCtx::MatchScrutineeExpr => "`match` scrutinee expression",
            UnusedDelimsCtx::ReturnValue => "`return` value",
            UnusedDelimsCtx::BlockRetValue => "block return value",
            UnusedDelimsCtx::BreakValue => "`break` value",
            UnusedDelimsCtx::LetScrutineeExpr => "`let` scrutinee expression",
            UnusedDelimsCtx::ArrayLenExpr | UnusedDelimsCtx::AnonConst => "const expression",
            UnusedDelimsCtx::MatchArmExpr => "match arm expression",
            UnusedDelimsCtx::IndexExpr => "index expression",
        }
    }
}

/// Used by both `UnusedParens` and `UnusedBraces` to prevent code duplication.
trait UnusedDelimLint {
    const DELIM_STR: &'static str;

    /// Due to `ref` pattern, there can be a difference between using
    /// `{ expr }` and `expr` in pattern-matching contexts. This means
    /// that we should only lint `unused_parens` and not `unused_braces`
    /// in this case.
    ///
    /// ```rust
    /// let mut a = 7;
    /// let ref b = { a }; // We actually borrow a copy of `a` here.
    /// a += 1; // By mutating `a` we invalidate any borrows of `a`.
    /// assert_eq!(b + 1, a); // `b` does not borrow `a`, so we can still use it here.
    /// ```
    const LINT_EXPR_IN_PATTERN_MATCHING_CTX: bool;

    // this cannot be a constant is it refers to a static.
    fn lint(&self) -> &'static Lint;

    fn check_unused_delims_expr(
        &self,
        cx: &EarlyContext<'_>,
        value: &ast::Expr,
        ctx: UnusedDelimsCtx,
        followed_by_block: bool,
        left_pos: Option<BytePos>,
        right_pos: Option<BytePos>,
        is_kw: bool,
    );

    fn is_expr_delims_necessary(
        inner: &ast::Expr,
        ctx: UnusedDelimsCtx,
        followed_by_block: bool,
    ) -> bool {
        let followed_by_else = ctx == UnusedDelimsCtx::AssignedValueLetElse;

        if followed_by_else {
            match inner.kind {
                ast::ExprKind::Binary(op, ..) if op.node.is_lazy() => return true,
                _ if classify::expr_trailing_brace(inner).is_some() => return true,
                _ => {}
            }
        }

        // Check it's range in LetScrutineeExpr
        if let ast::ExprKind::Range(..) = inner.kind
            && matches!(ctx, UnusedDelimsCtx::LetScrutineeExpr)
        {
            return true;
        }

        // Do not lint against parentheses around `&raw [const|mut] expr`.
        // These parentheses will have to be added e.g. when calling a method on the result of this
        // expression, and we want to avoid churn wrt adding and removing parentheses.
        if matches!(inner.kind, ast::ExprKind::AddrOf(ast::BorrowKind::Raw, ..)) {
            return true;
        }

        // Check if LHS needs parens to prevent false-positives in cases like
        // `fn x() -> u8 { ({ 0 } + 1) }`.
        //
        // FIXME: https://github.com/rust-lang/rust/issues/119426
        // The syntax tree in this code is from after macro expansion, so the
        // current implementation has both false negatives and false positives
        // related to expressions containing macros.
        //
        //     macro_rules! m1 {
        //         () => {
        //             1
        //         };
        //     }
        //
        //     fn f1() -> u8 {
        //         // Lint says parens are not needed, but they are.
        //         (m1! {} + 1)
        //     }
        //
        //     macro_rules! m2 {
        //         () => {
        //             loop { break 1; }
        //         };
        //     }
        //
        //     fn f2() -> u8 {
        //         // Lint says parens are needed, but they are not.
        //         (m2!() + 1)
        //     }
        {
            let mut innermost = inner;
            loop {
                innermost = match &innermost.kind {
                    ExprKind::Binary(_op, lhs, _rhs) => lhs,
                    ExprKind::Call(fn_, _params) => fn_,
                    ExprKind::Cast(expr, _ty) => expr,
                    ExprKind::Type(expr, _ty) => expr,
                    ExprKind::Index(base, _subscript, _) => base,
                    _ => break,
                };
                if !classify::expr_requires_semi_to_be_stmt(innermost) {
                    return true;
                }
            }
        }

        // Check if RHS needs parens to prevent false-positives in cases like `if (() == return)
        // {}`.
        if !followed_by_block {
            return false;
        }

        // Check if we need parens for `match &( Struct { field:  }) {}`.
        {
            let mut innermost = inner;
            loop {
                innermost = match &innermost.kind {
                    ExprKind::AddrOf(_, _, expr) => expr,
                    _ => {
                        if parser::contains_exterior_struct_lit(innermost) {
                            return true;
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        let mut innermost = inner;
        loop {
            innermost = match &innermost.kind {
                ExprKind::Unary(_op, expr) => expr,
                ExprKind::Binary(_op, _lhs, rhs) => rhs,
                ExprKind::AssignOp(_op, _lhs, rhs) => rhs,
                ExprKind::Assign(_lhs, rhs, _span) => rhs,

                ExprKind::Ret(_) | ExprKind::Yield(..) | ExprKind::Yeet(..) => return true,

                ExprKind::Break(_label, None) => return false,
                ExprKind::Break(_label, Some(break_expr)) => {
                    return matches!(break_expr.kind, ExprKind::Block(..));
                }

                ExprKind::Range(_lhs, Some(rhs), _limits) => {
                    return matches!(rhs.kind, ExprKind::Block(..));
                }

                _ => return parser::contains_exterior_struct_lit(inner),
            }
        }
    }

    fn emit_unused_delims_expr(
        &self,
        cx: &EarlyContext<'_>,
        value: &ast::Expr,
        ctx: UnusedDelimsCtx,
        left_pos: Option<BytePos>,
        right_pos: Option<BytePos>,
        is_kw: bool,
    ) {
        let span_with_attrs = match value.kind {
            ast::ExprKind::Block(ref block, None) if let [stmt] = block.stmts.as_slice() => {
                // For the statements with attributes, like `{ #[allow()] println!("Hello!") }`,
                // the span should contains the attributes, or the suggestion will remove them.
                if let Some(attr_lo) = stmt.attrs().iter().map(|attr| attr.span.lo()).min() {
                    stmt.span.with_lo(attr_lo)
                } else {
                    stmt.span
                }
            }
            ast::ExprKind::Paren(ref expr) => {
                // For the expr with attributes, like `let _ = (#[inline] || println!("Hello!"));`,
                // the span should contains the attributes, or the suggestion will remove them.
                if let Some(attr_lo) = expr.attrs.iter().map(|attr| attr.span.lo()).min() {
                    expr.span.with_lo(attr_lo)
                } else {
                    expr.span
                }
            }
            _ => return,
        };
        let spans = span_with_attrs
            .find_ancestor_inside(value.span)
            .map(|span| (value.span.with_hi(span.lo()), value.span.with_lo(span.hi())));
        let keep_space = (
            left_pos.is_some_and(|s| s >= value.span.lo()),
            right_pos.is_some_and(|s| s <= value.span.hi()),
        );
        self.emit_unused_delims(cx, value.span, spans, ctx.into(), keep_space, is_kw);
    }

    fn emit_unused_delims(
        &self,
        cx: &EarlyContext<'_>,
        value_span: Span,
        spans: Option<(Span, Span)>,
        msg: &str,
        keep_space: (bool, bool),
        is_kw: bool,
    ) {
        let primary_span = if let Some((lo, hi)) = spans {
            if hi.is_empty() {
                // do not point at delims that do not exist
                return;
            }
            MultiSpan::from(vec![lo, hi])
        } else {
            MultiSpan::from(value_span)
        };
        let suggestion = spans.map(|(lo, hi)| {
            let sm = cx.sess().source_map();
            let lo_replace = if (keep_space.0 || is_kw)
                && let Ok(snip) = sm.span_to_prev_source(lo)
                && !snip.ends_with(' ')
            {
                " "
            } else {
                ""
            };

            let hi_replace = if keep_space.1
                && let Ok(snip) = sm.span_to_next_source(hi)
                && !snip.starts_with(' ')
            {
                " "
            } else {
                ""
            };
            UnusedDelimSuggestion {
                start_span: lo,
                start_replace: lo_replace,
                end_span: hi,
                end_replace: hi_replace,
            }
        });
        cx.emit_span_lint(
            self.lint(),
            primary_span,
            UnusedDelim { delim: Self::DELIM_STR, item: msg, suggestion },
        );
    }

    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &ast::Expr) {
        use rustc_ast::ExprKind::*;
        let (value, ctx, followed_by_block, left_pos, right_pos, is_kw) = match e.kind {
            // Do not lint `unused_braces` in `if let` expressions.
            If(ref cond, ref block, _)
                if !matches!(cond.kind, Let(..)) || Self::LINT_EXPR_IN_PATTERN_MATCHING_CTX =>
            {
                let left = e.span.lo() + rustc_span::BytePos(2);
                let right = block.span.lo();
                (cond, UnusedDelimsCtx::IfCond, true, Some(left), Some(right), true)
            }

            // Do not lint `unused_braces` in `while let` expressions.
            While(ref cond, ref block, ..)
                if !matches!(cond.kind, Let(..)) || Self::LINT_EXPR_IN_PATTERN_MATCHING_CTX =>
            {
                let left = e.span.lo() + rustc_span::BytePos(5);
                let right = block.span.lo();
                (cond, UnusedDelimsCtx::WhileCond, true, Some(left), Some(right), true)
            }

            ForLoop { ref iter, ref body, .. } => {
                (iter, UnusedDelimsCtx::ForIterExpr, true, None, Some(body.span.lo()), true)
            }

            Match(ref head, _, ast::MatchKind::Prefix)
                if Self::LINT_EXPR_IN_PATTERN_MATCHING_CTX =>
            {
                let left = e.span.lo() + rustc_span::BytePos(5);
                (head, UnusedDelimsCtx::MatchScrutineeExpr, true, Some(left), None, true)
            }

            Ret(Some(ref value)) => {
                let left = e.span.lo() + rustc_span::BytePos(3);
                (value, UnusedDelimsCtx::ReturnValue, false, Some(left), None, true)
            }

            Break(_, Some(ref value)) => {
                (value, UnusedDelimsCtx::BreakValue, false, None, None, true)
            }

            Index(_, ref value, _) => (value, UnusedDelimsCtx::IndexExpr, false, None, None, false),

            Assign(_, ref value, _) | AssignOp(.., ref value) => {
                (value, UnusedDelimsCtx::AssignedValue, false, None, None, false)
            }
            // either function/method call, or something this lint doesn't care about
            ref call_or_other => {
                let (args_to_check, ctx) = match *call_or_other {
                    Call(_, ref args) => (&args[..], UnusedDelimsCtx::FunctionArg),
                    MethodCall(ref call) => (&call.args[..], UnusedDelimsCtx::MethodArg),
                    // actual catch-all arm
                    _ => {
                        return;
                    }
                };
                // Don't lint if this is a nested macro expansion: otherwise, the lint could
                // trigger in situations that macro authors shouldn't have to care about, e.g.,
                // when a parenthesized token tree matched in one macro expansion is matched as
                // an expression in another and used as a fn/method argument (Issue #47775)
                if e.span.ctxt().outer_expn_data().call_site.from_expansion() {
                    return;
                }
                for arg in args_to_check {
                    self.check_unused_delims_expr(cx, arg, ctx, false, None, None, false);
                }
                return;
            }
        };
        self.check_unused_delims_expr(
            cx,
            value,
            ctx,
            followed_by_block,
            left_pos,
            right_pos,
            is_kw,
        );
    }

    fn check_stmt(&mut self, cx: &EarlyContext<'_>, s: &ast::Stmt) {
        match s.kind {
            StmtKind::Let(ref local) if Self::LINT_EXPR_IN_PATTERN_MATCHING_CTX => {
                if let Some((init, els)) = local.kind.init_else_opt() {
                    if els.is_some()
                        && let ExprKind::Paren(paren) = &init.kind
                        && !init.span.eq_ctxt(paren.span)
                    {
                        // This branch prevents cases where parentheses wrap an expression
                        // resulting from macro expansion, such as:
                        // ```
                        // macro_rules! x {
                        // () => { None::<i32> };
                        // }
                        // let Some(_) = (x!{}) else { return };
                        // // -> let Some(_) = (None::<i32>) else { return };
                        // //                  ~           ~ No Lint
                        // ```
                        return;
                    }
                    let ctx = match els {
                        None => UnusedDelimsCtx::AssignedValue,
                        Some(_) => UnusedDelimsCtx::AssignedValueLetElse,
                    };
                    self.check_unused_delims_expr(cx, init, ctx, false, None, None, false);
                }
            }
            StmtKind::Expr(ref expr) => {
                self.check_unused_delims_expr(
                    cx,
                    expr,
                    UnusedDelimsCtx::BlockRetValue,
                    false,
                    None,
                    None,
                    false,
                );
            }
            _ => {}
        }
    }

    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &ast::Item) {
        use ast::ItemKind::*;

        if let Const(box ast::ConstItem { expr: Some(expr), .. })
        | Static(box ast::StaticItem { expr: Some(expr), .. }) = &item.kind
        {
            self.check_unused_delims_expr(
                cx,
                expr,
                UnusedDelimsCtx::AssignedValue,
                false,
                None,
                None,
                false,
            );
        }
    }
}

declare_lint! {
    /// The `unused_parens` lint detects `if`, `match`, `while` and `return`
    /// with parentheses; they do not need them.
    ///
    /// ### Examples
    ///
    /// ```rust
    /// if(true) {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The parentheses are not needed, and should be removed. This is the
    /// preferred style for writing these expressions.
    pub(super) UNUSED_PARENS,
    Warn,
    "`if`, `match`, `while` and `return` do not need parentheses"
}

#[derive(Default)]
pub(crate) struct UnusedParens {
    with_self_ty_parens: bool,
    /// `1 as (i32) < 2` parses to ExprKind::Lt
    /// `1 as i32 < 2` parses to i32::<2[missing angle bracket]
    parens_in_cast_in_lt: Vec<ast::NodeId>,
}

impl_lint_pass!(UnusedParens => [UNUSED_PARENS]);

impl UnusedDelimLint for UnusedParens {
    const DELIM_STR: &'static str = "parentheses";

    const LINT_EXPR_IN_PATTERN_MATCHING_CTX: bool = true;

    fn lint(&self) -> &'static Lint {
        UNUSED_PARENS
    }

    fn check_unused_delims_expr(
        &self,
        cx: &EarlyContext<'_>,
        value: &ast::Expr,
        ctx: UnusedDelimsCtx,
        followed_by_block: bool,
        left_pos: Option<BytePos>,
        right_pos: Option<BytePos>,
        is_kw: bool,
    ) {
        match value.kind {
            ast::ExprKind::Paren(ref inner) => {
                if !Self::is_expr_delims_necessary(inner, ctx, followed_by_block)
                    && value.attrs.is_empty()
                    && !value.span.from_expansion()
                    && (ctx != UnusedDelimsCtx::LetScrutineeExpr
                        || !matches!(inner.kind, ast::ExprKind::Binary(
                                rustc_span::source_map::Spanned { node, .. },
                                _,
                                _,
                            ) if node.is_lazy()))
                    && !((ctx == UnusedDelimsCtx::ReturnValue
                        || ctx == UnusedDelimsCtx::BreakValue)
                        && matches!(inner.kind, ast::ExprKind::Assign(_, _, _)))
                {
                    self.emit_unused_delims_expr(cx, value, ctx, left_pos, right_pos, is_kw)
                }
            }
            ast::ExprKind::Let(_, ref expr, _, _) => {
                self.check_unused_delims_expr(
                    cx,
                    expr,
                    UnusedDelimsCtx::LetScrutineeExpr,
                    followed_by_block,
                    None,
                    None,
                    false,
                );
            }
            _ => {}
        }
    }
}

impl UnusedParens {
    fn check_unused_parens_pat(
        &self,
        cx: &EarlyContext<'_>,
        value: &ast::Pat,
        avoid_or: bool,
        avoid_mut: bool,
        keep_space: (bool, bool),
    ) {
        use ast::{BindingMode, PatKind};

        if let PatKind::Paren(inner) = &value.kind {
            match inner.kind {
                // The lint visitor will visit each subpattern of `p`. We do not want to lint
                // any range pattern no matter where it occurs in the pattern. For something like
                // `&(a..=b)`, there is a recursive `check_pat` on `a` and `b`, but we will assume
                // that if there are unnecessary parens they serve a purpose of readability.
                PatKind::Range(..) => return,
                // Avoid `p0 | .. | pn` if we should.
                PatKind::Or(..) if avoid_or => return,
                // Avoid `mut x` and `mut x @ p` if we should:
                PatKind::Ident(BindingMode::MUT, ..) if avoid_mut => {
                    return;
                }
                // Otherwise proceed with linting.
                _ => {}
            }
            let spans = if !value.span.from_expansion() {
                inner
                    .span
                    .find_ancestor_inside(value.span)
                    .map(|inner| (value.span.with_hi(inner.lo()), value.span.with_lo(inner.hi())))
            } else {
                None
            };
            self.emit_unused_delims(cx, value.span, spans, "pattern", keep_space, false);
        }
    }

    fn cast_followed_by_lt(&self, expr: &ast::Expr) -> Option<ast::NodeId> {
        if let ExprKind::Binary(op, lhs, _rhs) = &expr.kind
            && (op.node == ast::BinOpKind::Lt || op.node == ast::BinOpKind::Shl)
        {
            let mut cur = lhs;
            while let ExprKind::Binary(_, _, rhs) = &cur.kind {
                cur = rhs;
            }

            if let ExprKind::Cast(_, ty) = &cur.kind
                && let ast::TyKind::Paren(_) = &ty.kind
            {
                return Some(ty.id);
            }
        }
        None
    }
}

impl EarlyLintPass for UnusedParens {
    #[inline]
    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &ast::Expr) {
        if let Some(ty_id) = self.cast_followed_by_lt(e) {
            self.parens_in_cast_in_lt.push(ty_id);
        }

        match e.kind {
            ExprKind::Let(ref pat, _, _, _) | ExprKind::ForLoop { ref pat, .. } => {
                self.check_unused_parens_pat(cx, pat, false, false, (true, true));
            }
            // We ignore parens in cases like `if (((let Some(0) = Some(1))))` because we already
            // handle a hard error for them during AST lowering in `lower_expr_mut`, but we still
            // want to complain about things like `if let 42 = (42)`.
            ExprKind::If(ref cond, ref block, ref else_)
                if matches!(cond.peel_parens().kind, ExprKind::Let(..)) =>
            {
                self.check_unused_delims_expr(
                    cx,
                    cond.peel_parens(),
                    UnusedDelimsCtx::LetScrutineeExpr,
                    true,
                    None,
                    None,
                    true,
                );
                for stmt in &block.stmts {
                    <Self as UnusedDelimLint>::check_stmt(self, cx, stmt);
                }
                if let Some(e) = else_ {
                    <Self as UnusedDelimLint>::check_expr(self, cx, e);
                }
                return;
            }
            ExprKind::Match(ref _expr, ref arm, _) => {
                for a in arm {
                    if let Some(body) = &a.body {
                        self.check_unused_delims_expr(
                            cx,
                            body,
                            UnusedDelimsCtx::MatchArmExpr,
                            false,
                            None,
                            None,
                            true,
                        );
                    }
                }
            }
            _ => {}
        }

        <Self as UnusedDelimLint>::check_expr(self, cx, e)
    }

    fn check_expr_post(&mut self, _cx: &EarlyContext<'_>, e: &ast::Expr) {
        if let Some(ty_id) = self.cast_followed_by_lt(e) {
            let id = self
                .parens_in_cast_in_lt
                .pop()
                .expect("check_expr and check_expr_post must balance");
            assert_eq!(
                id, ty_id,
                "check_expr, check_ty, and check_expr_post are called, in that order, by the visitor"
            );
        }
    }

    fn check_pat(&mut self, cx: &EarlyContext<'_>, p: &ast::Pat) {
        use ast::Mutability;
        use ast::PatKind::*;
        let keep_space = (false, false);
        match &p.kind {
            // Do not lint on `(..)` as that will result in the other arms being useless.
            Paren(_)
            // The other cases do not contain sub-patterns.
            | Missing | Wild | Never | Rest | Expr(..) | MacCall(..) | Range(..) | Ident(.., None)
            | Path(..) | Err(_) => {},
            // These are list-like patterns; parens can always be removed.
            TupleStruct(_, _, ps) | Tuple(ps) | Slice(ps) | Or(ps) => for p in ps {
                self.check_unused_parens_pat(cx, p, false, false, keep_space);
            },
            Struct(_, _, fps, _) => for f in fps {
                self.check_unused_parens_pat(cx, &f.pat, false, false, keep_space);
            },
            // Avoid linting on `i @ (p0 | .. | pn)` and `box (p0 | .. | pn)`, #64106.
            Ident(.., Some(p)) | Box(p) | Deref(p) | Guard(p, _) => self.check_unused_parens_pat(cx, p, true, false, keep_space),
            // Avoid linting on `&(mut x)` as `&mut x` has a different meaning, #55342.
            // Also avoid linting on `& mut? (p0 | .. | pn)`, #64106.
            Ref(p, m) => self.check_unused_parens_pat(cx, p, true, *m == Mutability::Not, keep_space),
        }
    }

    fn check_stmt(&mut self, cx: &EarlyContext<'_>, s: &ast::Stmt) {
        if let StmtKind::Let(ref local) = s.kind {
            self.check_unused_parens_pat(cx, &local.pat, true, false, (true, false));
        }

        <Self as UnusedDelimLint>::check_stmt(self, cx, s)
    }

    fn check_param(&mut self, cx: &EarlyContext<'_>, param: &ast::Param) {
        self.check_unused_parens_pat(cx, &param.pat, true, false, (false, false));
    }

    fn check_arm(&mut self, cx: &EarlyContext<'_>, arm: &ast::Arm) {
        self.check_unused_parens_pat(cx, &arm.pat, false, false, (false, false));
    }

    fn check_ty(&mut self, cx: &EarlyContext<'_>, ty: &ast::Ty) {
        if let ast::TyKind::Paren(_) = ty.kind
            && Some(&ty.id) == self.parens_in_cast_in_lt.last()
        {
            return;
        }
        match &ty.kind {
            ast::TyKind::Array(_, len) => {
                self.check_unused_delims_expr(
                    cx,
                    &len.value,
                    UnusedDelimsCtx::ArrayLenExpr,
                    false,
                    None,
                    None,
                    false,
                );
            }
            ast::TyKind::Paren(r) => {
                match &r.kind {
                    ast::TyKind::TraitObject(..) => {}
                    ast::TyKind::BareFn(b)
                        if self.with_self_ty_parens && b.generic_params.len() > 0 => {}
                    ast::TyKind::ImplTrait(_, bounds) if bounds.len() > 1 => {}
                    _ => {
                        let spans = if !ty.span.from_expansion() {
                            r.span
                                .find_ancestor_inside(ty.span)
                                .map(|r| (ty.span.with_hi(r.lo()), ty.span.with_lo(r.hi())))
                        } else {
                            None
                        };
                        self.emit_unused_delims(cx, ty.span, spans, "type", (false, false), false);
                    }
                }
                self.with_self_ty_parens = false;
            }
            _ => {}
        }
    }

    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &ast::Item) {
        <Self as UnusedDelimLint>::check_item(self, cx, item)
    }

    fn enter_where_predicate(&mut self, _: &EarlyContext<'_>, pred: &ast::WherePredicate) {
        use rustc_ast::{WhereBoundPredicate, WherePredicateKind};
        if let WherePredicateKind::BoundPredicate(WhereBoundPredicate {
            bounded_ty,
            bound_generic_params,
            ..
        }) = &pred.kind
            && let ast::TyKind::Paren(_) = &bounded_ty.kind
            && bound_generic_params.is_empty()
        {
            self.with_self_ty_parens = true;
        }
    }

    fn exit_where_predicate(&mut self, _: &EarlyContext<'_>, _: &ast::WherePredicate) {
        assert!(!self.with_self_ty_parens);
    }
}

declare_lint! {
    /// The `unused_braces` lint detects unnecessary braces around an
    /// expression.
    ///
    /// ### Example
    ///
    /// ```rust
    /// if { true } {
    ///     // ...
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The braces are not needed, and should be removed. This is the
    /// preferred style for writing these expressions.
    pub(super) UNUSED_BRACES,
    Warn,
    "unnecessary braces around an expression"
}

declare_lint_pass!(UnusedBraces => [UNUSED_BRACES]);

impl UnusedDelimLint for UnusedBraces {
    const DELIM_STR: &'static str = "braces";

    const LINT_EXPR_IN_PATTERN_MATCHING_CTX: bool = false;

    fn lint(&self) -> &'static Lint {
        UNUSED_BRACES
    }

    fn check_unused_delims_expr(
        &self,
        cx: &EarlyContext<'_>,
        value: &ast::Expr,
        ctx: UnusedDelimsCtx,
        followed_by_block: bool,
        left_pos: Option<BytePos>,
        right_pos: Option<BytePos>,
        is_kw: bool,
    ) {
        match value.kind {
            ast::ExprKind::Block(ref inner, None)
                if inner.rules == ast::BlockCheckMode::Default =>
            {
                // emit a warning under the following conditions:
                //
                // - the block does not have a label
                // - the block is not `unsafe`
                // - the block contains exactly one expression (do not lint `{ expr; }`)
                // - `followed_by_block` is true and the internal expr may contain a `{`
                // - the block is not multiline (do not lint multiline match arms)
                //      ```
                //      match expr {
                //          Pattern => {
                //              somewhat_long_expression
                //          }
                //          // ...
                //      }
                //      ```
                // - the block has no attribute and was not created inside a macro
                // - if the block is an `anon_const`, the inner expr must be a literal
                //   not created by a macro, i.e. do not lint on:
                //      ```
                //      struct A<const N: usize>;
                //      let _: A<{ 2 + 3 }>;
                //      let _: A<{produces_literal!()}>;
                //      ```
                // FIXME(const_generics): handle paths when #67075 is fixed.
                if let [stmt] = inner.stmts.as_slice() {
                    if let ast::StmtKind::Expr(ref expr) = stmt.kind {
                        if !Self::is_expr_delims_necessary(expr, ctx, followed_by_block)
                            && (ctx != UnusedDelimsCtx::AnonConst
                                || (matches!(expr.kind, ast::ExprKind::Lit(_))
                                    && !expr.span.from_expansion()))
                            && !cx.sess().source_map().is_multiline(value.span)
                            && value.attrs.is_empty()
                            && !value.span.from_expansion()
                            && !inner.span.from_expansion()
                        {
                            self.emit_unused_delims_expr(cx, value, ctx, left_pos, right_pos, is_kw)
                        }
                    }
                }
            }
            ast::ExprKind::Let(_, ref expr, _, _) => {
                self.check_unused_delims_expr(
                    cx,
                    expr,
                    UnusedDelimsCtx::LetScrutineeExpr,
                    followed_by_block,
                    None,
                    None,
                    false,
                );
            }
            _ => {}
        }
    }
}

impl EarlyLintPass for UnusedBraces {
    fn check_stmt(&mut self, cx: &EarlyContext<'_>, s: &ast::Stmt) {
        <Self as UnusedDelimLint>::check_stmt(self, cx, s)
    }

    #[inline]
    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &ast::Expr) {
        <Self as UnusedDelimLint>::check_expr(self, cx, e);

        if let ExprKind::Repeat(_, ref anon_const) = e.kind {
            self.check_unused_delims_expr(
                cx,
                &anon_const.value,
                UnusedDelimsCtx::AnonConst,
                false,
                None,
                None,
                false,
            );
        }
    }

    fn check_generic_arg(&mut self, cx: &EarlyContext<'_>, arg: &ast::GenericArg) {
        if let ast::GenericArg::Const(ct) = arg {
            self.check_unused_delims_expr(
                cx,
                &ct.value,
                UnusedDelimsCtx::AnonConst,
                false,
                None,
                None,
                false,
            );
        }
    }

    fn check_variant(&mut self, cx: &EarlyContext<'_>, v: &ast::Variant) {
        if let Some(anon_const) = &v.disr_expr {
            self.check_unused_delims_expr(
                cx,
                &anon_const.value,
                UnusedDelimsCtx::AnonConst,
                false,
                None,
                None,
                false,
            );
        }
    }

    fn check_ty(&mut self, cx: &EarlyContext<'_>, ty: &ast::Ty) {
        match ty.kind {
            ast::TyKind::Array(_, ref len) => {
                self.check_unused_delims_expr(
                    cx,
                    &len.value,
                    UnusedDelimsCtx::ArrayLenExpr,
                    false,
                    None,
                    None,
                    false,
                );
            }

            ast::TyKind::Typeof(ref anon_const) => {
                self.check_unused_delims_expr(
                    cx,
                    &anon_const.value,
                    UnusedDelimsCtx::AnonConst,
                    false,
                    None,
                    None,
                    false,
                );
            }

            _ => {}
        }
    }

    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &ast::Item) {
        <Self as UnusedDelimLint>::check_item(self, cx, item)
    }
}

declare_lint! {
    /// The `unused_import_braces` lint catches unnecessary braces around an
    /// imported item.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(unused_import_braces)]
    /// use test::{A};
    ///
    /// pub mod test {
    ///     pub struct A;
    /// }
    /// # fn main() {}
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// If there is only a single item, then remove the braces (`use test::A;`
    /// for example).
    ///
    /// This lint is "allow" by default because it is only enforcing a
    /// stylistic choice.
    UNUSED_IMPORT_BRACES,
    Allow,
    "unnecessary braces around an imported item"
}

declare_lint_pass!(UnusedImportBraces => [UNUSED_IMPORT_BRACES]);

impl UnusedImportBraces {
    fn check_use_tree(&self, cx: &EarlyContext<'_>, use_tree: &ast::UseTree, item: &ast::Item) {
        if let ast::UseTreeKind::Nested { ref items, .. } = use_tree.kind {
            // Recursively check nested UseTrees
            for (tree, _) in items {
                self.check_use_tree(cx, tree, item);
            }

            // Trigger the lint only if there is one nested item
            let [(tree, _)] = items.as_slice() else { return };

            // Trigger the lint if the nested item is a non-self single item
            let node_name = match tree.kind {
                ast::UseTreeKind::Simple(rename) => {
                    let orig_ident = tree.prefix.segments.last().unwrap().ident;
                    if orig_ident.name == kw::SelfLower {
                        return;
                    }
                    rename.unwrap_or(orig_ident).name
                }
                ast::UseTreeKind::Glob => sym::asterisk,
                ast::UseTreeKind::Nested { .. } => return,
            };

            cx.emit_span_lint(
                UNUSED_IMPORT_BRACES,
                item.span,
                UnusedImportBracesDiag { node: node_name },
            );
        }
    }
}

impl EarlyLintPass for UnusedImportBraces {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &ast::Item) {
        if let ast::ItemKind::Use(ref use_tree) = item.kind {
            self.check_use_tree(cx, use_tree, item);
        }
    }
}

declare_lint! {
    /// The `unused_allocation` lint detects unnecessary allocations that can
    /// be eliminated.
    ///
    /// ### Example
    ///
    /// ```rust
    /// fn main() {
    ///     let a = Box::new([1, 2, 3]).len();
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// When a `box` expression is immediately coerced to a reference, then
    /// the allocation is unnecessary, and a reference (using `&` or `&mut`)
    /// should be used instead to avoid the allocation.
    pub(super) UNUSED_ALLOCATION,
    Warn,
    "detects unnecessary allocations that can be eliminated"
}

declare_lint_pass!(UnusedAllocation => [UNUSED_ALLOCATION]);

impl<'tcx> LateLintPass<'tcx> for UnusedAllocation {
    fn check_expr(&mut self, cx: &LateContext<'_>, e: &hir::Expr<'_>) {
        match e.kind {
            hir::ExprKind::Call(path_expr, [_])
                if let hir::ExprKind::Path(qpath) = &path_expr.kind
                    && let Some(did) = cx.qpath_res(qpath, path_expr.hir_id).opt_def_id()
                    && cx.tcx.is_diagnostic_item(sym::box_new, did) => {}
            _ => return,
        }

        for adj in cx.typeck_results().expr_adjustments(e) {
            if let adjustment::Adjust::Borrow(adjustment::AutoBorrow::Ref(m)) = adj.kind {
                match m {
                    adjustment::AutoBorrowMutability::Not => {
                        cx.emit_span_lint(UNUSED_ALLOCATION, e.span, UnusedAllocationDiag);
                    }
                    adjustment::AutoBorrowMutability::Mut { .. } => {
                        cx.emit_span_lint(UNUSED_ALLOCATION, e.span, UnusedAllocationMutDiag);
                    }
                };
            }
        }
    }
}
