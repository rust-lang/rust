use crate::lints::{
    PathStatementDrop, PathStatementDropSub, PathStatementNoEffect, UnusedAllocationDiag,
    UnusedAllocationMutDiag, UnusedClosure, UnusedDef, UnusedDefSuggestion, UnusedDelim,
    UnusedDelimSuggestion, UnusedGenerator, UnusedImportBracesDiag, UnusedOp, UnusedResult,
};
use crate::Lint;
use crate::{EarlyContext, EarlyLintPass, LateContext, LateLintPass, LintContext};
use rustc_ast as ast;
use rustc_ast::util::{classify, parser};
use rustc_ast::{ExprKind, StmtKind};
use rustc_errors::{pluralize, MultiSpan};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_infer::traits::util::elaborate_predicates_with_span;
use rustc_middle::ty::adjustment;
use rustc_middle::ty::{self, DefIdTree, Ty};
use rustc_span::symbol::Symbol;
use rustc_span::symbol::{kw, sym};
use rustc_span::{BytePos, Span};
use std::iter;

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
        let hir::StmtKind::Semi(expr) = s.kind else { return; };

        if let hir::ExprKind::Ret(..) = expr.kind {
            return;
        }

        if let hir::ExprKind::Match(await_expr, _arms, hir::MatchSource::AwaitDesugar) = expr.kind
            && let ty = cx.typeck_results().expr_ty(&await_expr)
            && let ty::Alias(ty::Opaque, ty::AliasTy { def_id: future_def_id, .. }) = ty.kind()
            && cx.tcx.ty_is_opaque_future(ty)
            // FIXME: This also includes non-async fns that return `impl Future`.
            && let async_fn_def_id = cx.tcx.parent(*future_def_id)
            && check_must_use_def(
                cx,
                async_fn_def_id,
                expr.span,
                "output of future returned by ",
                "",
            )
        {
            // We have a bare `foo().await;` on an opaque type from an async function that was
            // annotated with `#[must_use]`.
            return;
        }

        let ty = cx.typeck_results().expr_ty(&expr);

        let must_use_result = is_ty_must_use(cx, ty, &expr, expr.span);
        let type_lint_emitted_or_suppressed = match must_use_result {
            Some(path) => {
                emit_must_use_untranslated(cx, &path, "", "", 1);
                true
            }
            None => false,
        };

        let fn_warned = check_fn_must_use(cx, expr);

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
            hir::ExprKind::Unary(..) => Some("unary operation"),
            _ => None,
        };

        let mut op_warned = false;

        if let Some(must_use_op) = must_use_op {
            cx.emit_spanned_lint(
                UNUSED_MUST_USE,
                expr.span,
                UnusedOp {
                    op: must_use_op,
                    label: expr.span,
                    suggestion: expr.span.shrink_to_lo(),
                },
            );
            op_warned = true;
        }

        if !(type_lint_emitted_or_suppressed || fn_warned || op_warned) {
            cx.emit_spanned_lint(UNUSED_RESULTS, s.span, UnusedResult { ty });
        }

        fn check_fn_must_use(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
            let maybe_def_id = match expr.kind {
                hir::ExprKind::Call(ref callee, _) => {
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
                check_must_use_def(cx, def_id, expr.span, "return value of ", "")
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
            Opaque(Box<Self>),
            TraitObject(Box<Self>),
            TupleElement(Vec<(usize, Self)>),
            Array(Box<Self>, u64),
            /// The root of the unused_closures lint.
            Closure(Span),
            /// The root of the unused_generators lint.
            Generator(Span),
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
                    cx.param_env,
                )
            {
                return Some(MustUsePath::Suppressed);
            }

            match *ty.kind() {
                ty::Adt(..) if ty.is_box() => {
                    let boxed_ty = ty.boxed_ty();
                    is_ty_must_use(cx, boxed_ty, expr, span)
                        .map(|inner| MustUsePath::Boxed(Box::new(inner)))
                }
                ty::Adt(def, _) => is_def_must_use(cx, def.did(), span),
                ty::Alias(ty::Opaque, ty::AliasTy { def_id: def, .. }) => {
                    elaborate_predicates_with_span(
                        cx.tcx,
                        cx.tcx.explicit_item_bounds(def).iter().cloned(),
                    )
                    .find_map(|obligation| {
                        // We only look at the `DefId`, so it is safe to skip the binder here.
                        if let ty::PredicateKind::Clause(ty::Clause::Trait(
                            ref poly_trait_predicate,
                        )) = obligation.predicate.kind().skip_binder()
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
                ty::Array(ty, len) => match len.try_eval_target_usize(cx.tcx, cx.param_env) {
                    // If the array is empty we don't lint, to avoid false positives
                    Some(0) | None => None,
                    // If the array is definitely non-empty, we can do `#[must_use]` checking.
                    Some(len) => is_ty_must_use(cx, ty, expr, span)
                        .map(|inner| MustUsePath::Array(Box::new(inner), len)),
                },
                ty::Closure(..) => Some(MustUsePath::Closure(span)),
                ty::Generator(def_id, ..) => {
                    // async fn should be treated as "implementor of `Future`"
                    let must_use = if cx.tcx.generator_is_async(def_id) {
                        let def_id = cx.tcx.lang_items().future_trait().unwrap();
                        is_def_must_use(cx, def_id, span)
                            .map(|inner| MustUsePath::Opaque(Box::new(inner)))
                    } else {
                        None
                    };
                    must_use.or(Some(MustUsePath::Generator(span)))
                }
                _ => None,
            }
        }

        fn is_def_must_use(cx: &LateContext<'_>, def_id: DefId, span: Span) -> Option<MustUsePath> {
            if let Some(attr) = cx.tcx.get_attr(def_id, sym::must_use) {
                // check for #[must_use = "..."]
                let reason = attr.value_str();
                Some(MustUsePath::Def(span, def_id, reason))
            } else {
                None
            }
        }

        // Returns whether further errors should be suppressed because either a lint has been emitted or the type should be ignored.
        fn check_must_use_def(
            cx: &LateContext<'_>,
            def_id: DefId,
            span: Span,
            descr_pre_path: &str,
            descr_post_path: &str,
        ) -> bool {
            is_def_must_use(cx, def_id, span)
                .map(|must_use_path| {
                    emit_must_use_untranslated(
                        cx,
                        &must_use_path,
                        descr_pre_path,
                        descr_post_path,
                        1,
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
        ) {
            let plural_suffix = pluralize!(plural_len);

            match path {
                MustUsePath::Suppressed => {}
                MustUsePath::Boxed(path) => {
                    let descr_pre = &format!("{}boxed ", descr_pre);
                    emit_must_use_untranslated(cx, path, descr_pre, descr_post, plural_len);
                }
                MustUsePath::Opaque(path) => {
                    let descr_pre = &format!("{}implementer{} of ", descr_pre, plural_suffix);
                    emit_must_use_untranslated(cx, path, descr_pre, descr_post, plural_len);
                }
                MustUsePath::TraitObject(path) => {
                    let descr_post = &format!(" trait object{}{}", plural_suffix, descr_post);
                    emit_must_use_untranslated(cx, path, descr_pre, descr_post, plural_len);
                }
                MustUsePath::TupleElement(elems) => {
                    for (index, path) in elems {
                        let descr_post = &format!(" in tuple element {}", index);
                        emit_must_use_untranslated(cx, path, descr_pre, descr_post, plural_len);
                    }
                }
                MustUsePath::Array(path, len) => {
                    let descr_pre = &format!("{}array{} of ", descr_pre, plural_suffix);
                    emit_must_use_untranslated(
                        cx,
                        path,
                        descr_pre,
                        descr_post,
                        plural_len.saturating_add(usize::try_from(*len).unwrap_or(usize::MAX)),
                    );
                }
                MustUsePath::Closure(span) => {
                    cx.emit_spanned_lint(
                        UNUSED_MUST_USE,
                        *span,
                        UnusedClosure { count: plural_len, pre: descr_pre, post: descr_post },
                    );
                }
                MustUsePath::Generator(span) => {
                    cx.emit_spanned_lint(
                        UNUSED_MUST_USE,
                        *span,
                        UnusedGenerator { count: plural_len, pre: descr_pre, post: descr_post },
                    );
                }
                MustUsePath::Def(span, def_id, reason) => {
                    let suggestion = if matches!(
                        cx.tcx.get_diagnostic_name(*def_id),
                        Some(sym::add)
                            | Some(sym::sub)
                            | Some(sym::mul)
                            | Some(sym::div)
                            | Some(sym::rem)
                            | Some(sym::neg),
                    ) {
                        Some(UnusedDefSuggestion::Default { span: span.shrink_to_lo() })
                    } else {
                        None
                    };
                    cx.emit_spanned_lint(
                        UNUSED_MUST_USE,
                        *span,
                        UnusedDef {
                            pre: descr_pre,
                            post: descr_post,
                            cx,
                            def_id: *def_id,
                            note: *reason,
                            suggestion,
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
                if ty.needs_drop(cx.tcx, cx.param_env) {
                    let sub = if let Ok(snippet) = cx.sess().source_map().span_to_snippet(expr.span)
                    {
                        PathStatementDropSub::Suggestion { span: s.span, snippet }
                    } else {
                        PathStatementDropSub::Help { span: s.span }
                    };
                    cx.emit_spanned_lint(PATH_STATEMENTS, s.span, PathStatementDrop { sub })
                } else {
                    cx.emit_spanned_lint(PATH_STATEMENTS, s.span, PathStatementNoEffect);
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
    );

    fn is_expr_delims_necessary(
        inner: &ast::Expr,
        followed_by_block: bool,
        followed_by_else: bool,
    ) -> bool {
        if followed_by_else {
            match inner.kind {
                ast::ExprKind::Binary(op, ..) if op.node.lazy() => return true,
                _ if classify::expr_trailing_brace(inner).is_some() => return true,
                _ => {}
            }
        }

        // Prevent false-positives in cases like `fn x() -> u8 { ({ 0 } + 1) }`
        let lhs_needs_parens = {
            let mut innermost = inner;
            loop {
                innermost = match &innermost.kind {
                    ExprKind::Binary(_, lhs, _rhs) => lhs,
                    ExprKind::Call(fn_, _params) => fn_,
                    ExprKind::Cast(expr, _ty) => expr,
                    ExprKind::Type(expr, _ty) => expr,
                    ExprKind::Index(base, _subscript) => base,
                    _ => break false,
                };
                if !classify::expr_requires_semi_to_be_stmt(innermost) {
                    break true;
                }
            }
        };

        lhs_needs_parens
            || (followed_by_block
                && match &inner.kind {
                    ExprKind::Ret(_)
                    | ExprKind::Break(..)
                    | ExprKind::Yield(..)
                    | ExprKind::Yeet(..) => true,
                    ExprKind::Range(_lhs, Some(rhs), _limits) => {
                        matches!(rhs.kind, ExprKind::Block(..))
                    }
                    _ => parser::contains_exterior_struct_lit(&inner),
                })
    }

    fn emit_unused_delims_expr(
        &self,
        cx: &EarlyContext<'_>,
        value: &ast::Expr,
        ctx: UnusedDelimsCtx,
        left_pos: Option<BytePos>,
        right_pos: Option<BytePos>,
    ) {
        // If `value` has `ExprKind::Err`, unused delim lint can be broken.
        // For example, the following code caused ICE.
        // This is because the `ExprKind::Call` in `value` has `ExprKind::Err` as its argument
        // and this leads to wrong spans. #104897
        //
        // ```
        // fn f(){(print!(á
        // ```
        use rustc_ast::visit::{walk_expr, Visitor};
        struct ErrExprVisitor {
            has_error: bool,
        }
        impl<'ast> Visitor<'ast> for ErrExprVisitor {
            fn visit_expr(&mut self, expr: &'ast ast::Expr) {
                if let ExprKind::Err = expr.kind {
                    self.has_error = true;
                    return;
                }
                walk_expr(self, expr)
            }
        }
        let mut visitor = ErrExprVisitor { has_error: false };
        visitor.visit_expr(value);
        if visitor.has_error {
            return;
        }
        let spans = match value.kind {
            ast::ExprKind::Block(ref block, None) if block.stmts.len() == 1 => {
                if let Some(span) = block.stmts[0].span.find_ancestor_inside(value.span) {
                    Some((value.span.with_hi(span.lo()), value.span.with_lo(span.hi())))
                } else {
                    None
                }
            }
            ast::ExprKind::Paren(ref expr) => {
                let expr_span = expr.span.find_ancestor_inside(value.span);
                if let Some(expr_span) = expr_span {
                    Some((value.span.with_hi(expr_span.lo()), value.span.with_lo(expr_span.hi())))
                } else {
                    None
                }
            }
            _ => return,
        };
        let keep_space = (
            left_pos.map_or(false, |s| s >= value.span.lo()),
            right_pos.map_or(false, |s| s <= value.span.hi()),
        );
        self.emit_unused_delims(cx, value.span, spans, ctx.into(), keep_space);
    }

    fn emit_unused_delims(
        &self,
        cx: &EarlyContext<'_>,
        value_span: Span,
        spans: Option<(Span, Span)>,
        msg: &str,
        keep_space: (bool, bool),
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
            let lo_replace =
                    if keep_space.0 &&
                        let Ok(snip) = sm.span_to_prev_source(lo) && !snip.ends_with(' ') {
                        " "
                        } else {
                            ""
                        };

            let hi_replace =
                    if keep_space.1 &&
                        let Ok(snip) = sm.span_to_next_source(hi) && !snip.starts_with(' ') {
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
        cx.emit_spanned_lint(
            self.lint(),
            primary_span,
            UnusedDelim { delim: Self::DELIM_STR, item: msg, suggestion },
        );
    }

    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &ast::Expr) {
        use rustc_ast::ExprKind::*;
        let (value, ctx, followed_by_block, left_pos, right_pos) = match e.kind {
            // Do not lint `unused_braces` in `if let` expressions.
            If(ref cond, ref block, _)
                if !matches!(cond.kind, Let(_, _, _))
                    || Self::LINT_EXPR_IN_PATTERN_MATCHING_CTX =>
            {
                let left = e.span.lo() + rustc_span::BytePos(2);
                let right = block.span.lo();
                (cond, UnusedDelimsCtx::IfCond, true, Some(left), Some(right))
            }

            // Do not lint `unused_braces` in `while let` expressions.
            While(ref cond, ref block, ..)
                if !matches!(cond.kind, Let(_, _, _))
                    || Self::LINT_EXPR_IN_PATTERN_MATCHING_CTX =>
            {
                let left = e.span.lo() + rustc_span::BytePos(5);
                let right = block.span.lo();
                (cond, UnusedDelimsCtx::WhileCond, true, Some(left), Some(right))
            }

            ForLoop(_, ref cond, ref block, ..) => {
                (cond, UnusedDelimsCtx::ForIterExpr, true, None, Some(block.span.lo()))
            }

            Match(ref head, _) if Self::LINT_EXPR_IN_PATTERN_MATCHING_CTX => {
                let left = e.span.lo() + rustc_span::BytePos(5);
                (head, UnusedDelimsCtx::MatchScrutineeExpr, true, Some(left), None)
            }

            Ret(Some(ref value)) => {
                let left = e.span.lo() + rustc_span::BytePos(3);
                (value, UnusedDelimsCtx::ReturnValue, false, Some(left), None)
            }

            Index(_, ref value) => (value, UnusedDelimsCtx::IndexExpr, false, None, None),

            Assign(_, ref value, _) | AssignOp(.., ref value) => {
                (value, UnusedDelimsCtx::AssignedValue, false, None, None)
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
                    self.check_unused_delims_expr(cx, arg, ctx, false, None, None);
                }
                return;
            }
        };
        self.check_unused_delims_expr(cx, &value, ctx, followed_by_block, left_pos, right_pos);
    }

    fn check_stmt(&mut self, cx: &EarlyContext<'_>, s: &ast::Stmt) {
        match s.kind {
            StmtKind::Local(ref local) if Self::LINT_EXPR_IN_PATTERN_MATCHING_CTX => {
                if let Some((init, els)) = local.kind.init_else_opt() {
                    let ctx = match els {
                        None => UnusedDelimsCtx::AssignedValue,
                        Some(_) => UnusedDelimsCtx::AssignedValueLetElse,
                    };
                    self.check_unused_delims_expr(cx, init, ctx, false, None, None);
                }
            }
            StmtKind::Expr(ref expr) => {
                self.check_unused_delims_expr(
                    cx,
                    &expr,
                    UnusedDelimsCtx::BlockRetValue,
                    false,
                    None,
                    None,
                );
            }
            _ => {}
        }
    }

    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &ast::Item) {
        use ast::ItemKind::*;

        if let Const(.., Some(expr)) | Static(.., Some(expr)) = &item.kind {
            self.check_unused_delims_expr(
                cx,
                expr,
                UnusedDelimsCtx::AssignedValue,
                false,
                None,
                None,
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

pub struct UnusedParens {
    with_self_ty_parens: bool,
}

impl UnusedParens {
    pub fn new() -> Self {
        Self { with_self_ty_parens: false }
    }
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
    ) {
        match value.kind {
            ast::ExprKind::Paren(ref inner) => {
                let followed_by_else = ctx == UnusedDelimsCtx::AssignedValueLetElse;
                if !Self::is_expr_delims_necessary(inner, followed_by_block, followed_by_else)
                    && value.attrs.is_empty()
                    && !value.span.from_expansion()
                    && (ctx != UnusedDelimsCtx::LetScrutineeExpr
                        || !matches!(inner.kind, ast::ExprKind::Binary(
                                rustc_span::source_map::Spanned { node, .. },
                                _,
                                _,
                            ) if node.lazy()))
                {
                    self.emit_unused_delims_expr(cx, value, ctx, left_pos, right_pos)
                }
            }
            ast::ExprKind::Let(_, ref expr, _) => {
                self.check_unused_delims_expr(
                    cx,
                    expr,
                    UnusedDelimsCtx::LetScrutineeExpr,
                    followed_by_block,
                    None,
                    None,
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
        use ast::{BindingAnnotation, PatKind};

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
                PatKind::Ident(BindingAnnotation::MUT, ..) if avoid_mut => {
                    return;
                }
                // Otherwise proceed with linting.
                _ => {}
            }
            let spans = if let Some(inner) = inner.span.find_ancestor_inside(value.span) {
                Some((value.span.with_hi(inner.lo()), value.span.with_lo(inner.hi())))
            } else {
                None
            };
            self.emit_unused_delims(cx, value.span, spans, "pattern", keep_space);
        }
    }
}

impl EarlyLintPass for UnusedParens {
    #[inline]
    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &ast::Expr) {
        match e.kind {
            ExprKind::Let(ref pat, _, _) | ExprKind::ForLoop(ref pat, ..) => {
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
                );
                for stmt in &block.stmts {
                    <Self as UnusedDelimLint>::check_stmt(self, cx, stmt);
                }
                if let Some(e) = else_ {
                    <Self as UnusedDelimLint>::check_expr(self, cx, e);
                }
                return;
            }
            ExprKind::Match(ref _expr, ref arm) => {
                for a in arm {
                    self.check_unused_delims_expr(
                        cx,
                        &a.body,
                        UnusedDelimsCtx::MatchArmExpr,
                        false,
                        None,
                        None,
                    );
                }
            }
            _ => {}
        }

        <Self as UnusedDelimLint>::check_expr(self, cx, e)
    }

    fn check_pat(&mut self, cx: &EarlyContext<'_>, p: &ast::Pat) {
        use ast::{Mutability, PatKind::*};
        let keep_space = (false, false);
        match &p.kind {
            // Do not lint on `(..)` as that will result in the other arms being useless.
            Paren(_)
            // The other cases do not contain sub-patterns.
            | Wild | Rest | Lit(..) | MacCall(..) | Range(..) | Ident(.., None) | Path(..) => {},
            // These are list-like patterns; parens can always be removed.
            TupleStruct(_, _, ps) | Tuple(ps) | Slice(ps) | Or(ps) => for p in ps {
                self.check_unused_parens_pat(cx, p, false, false, keep_space);
            },
            Struct(_, _, fps, _) => for f in fps {
                self.check_unused_parens_pat(cx, &f.pat, false, false, keep_space);
            },
            // Avoid linting on `i @ (p0 | .. | pn)` and `box (p0 | .. | pn)`, #64106.
            Ident(.., Some(p)) | Box(p) => self.check_unused_parens_pat(cx, p, true, false, keep_space),
            // Avoid linting on `&(mut x)` as `&mut x` has a different meaning, #55342.
            // Also avoid linting on `& mut? (p0 | .. | pn)`, #64106.
            Ref(p, m) => self.check_unused_parens_pat(cx, p, true, *m == Mutability::Not, keep_space),
        }
    }

    fn check_stmt(&mut self, cx: &EarlyContext<'_>, s: &ast::Stmt) {
        if let StmtKind::Local(ref local) = s.kind {
            self.check_unused_parens_pat(cx, &local.pat, true, false, (false, false));
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
        match &ty.kind {
            ast::TyKind::Array(_, len) => {
                self.check_unused_delims_expr(
                    cx,
                    &len.value,
                    UnusedDelimsCtx::ArrayLenExpr,
                    false,
                    None,
                    None,
                );
            }
            ast::TyKind::Paren(r) => {
                match &r.kind {
                    ast::TyKind::TraitObject(..) => {}
                    ast::TyKind::BareFn(b)
                        if self.with_self_ty_parens && b.generic_params.len() > 0 => {}
                    ast::TyKind::ImplTrait(_, bounds) if bounds.len() > 1 => {}
                    _ => {
                        let spans = if let Some(r) = r.span.find_ancestor_inside(ty.span) {
                            Some((ty.span.with_hi(r.lo()), ty.span.with_lo(r.hi())))
                        } else {
                            None
                        };
                        self.emit_unused_delims(cx, ty.span, spans, "type", (false, false));
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
        use rustc_ast::{WhereBoundPredicate, WherePredicate};
        if let WherePredicate::BoundPredicate(WhereBoundPredicate {
                bounded_ty,
                bound_generic_params,
                ..
            }) = pred &&
            let ast::TyKind::Paren(_) = &bounded_ty.kind &&
            bound_generic_params.is_empty() {
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
                        if !Self::is_expr_delims_necessary(expr, followed_by_block, false)
                            && (ctx != UnusedDelimsCtx::AnonConst
                                || (matches!(expr.kind, ast::ExprKind::Lit(_))
                                    && !expr.span.from_expansion()))
                            && !cx.sess().source_map().is_multiline(value.span)
                            && value.attrs.is_empty()
                            && !value.span.from_expansion()
                            && !inner.span.from_expansion()
                        {
                            self.emit_unused_delims_expr(cx, value, ctx, left_pos, right_pos)
                        }
                    }
                }
            }
            ast::ExprKind::Let(_, ref expr, _) => {
                self.check_unused_delims_expr(
                    cx,
                    expr,
                    UnusedDelimsCtx::LetScrutineeExpr,
                    followed_by_block,
                    None,
                    None,
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
        if let ast::UseTreeKind::Nested(ref items) = use_tree.kind {
            // Recursively check nested UseTrees
            for (tree, _) in items {
                self.check_use_tree(cx, tree, item);
            }

            // Trigger the lint only if there is one nested item
            if items.len() != 1 {
                return;
            }

            // Trigger the lint if the nested item is a non-self single item
            let node_name = match items[0].0.kind {
                ast::UseTreeKind::Simple(rename) => {
                    let orig_ident = items[0].0.prefix.segments.last().unwrap().ident;
                    if orig_ident.name == kw::SelfLower {
                        return;
                    }
                    rename.unwrap_or(orig_ident).name
                }
                ast::UseTreeKind::Glob => Symbol::intern("*"),
                ast::UseTreeKind::Nested(_) => return,
            };

            cx.emit_spanned_lint(
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
    /// #![feature(box_syntax)]
    /// fn main() {
    ///     let a = (box [1, 2, 3]).len();
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
            hir::ExprKind::Box(_) => {}
            _ => return,
        }

        for adj in cx.typeck_results().expr_adjustments(e) {
            if let adjustment::Adjust::Borrow(adjustment::AutoBorrow::Ref(_, m)) = adj.kind {
                match m {
                    adjustment::AutoBorrowMutability::Not => {
                        cx.emit_spanned_lint(UNUSED_ALLOCATION, e.span, UnusedAllocationDiag);
                    }
                    adjustment::AutoBorrowMutability::Mut { .. } => {
                        cx.emit_spanned_lint(UNUSED_ALLOCATION, e.span, UnusedAllocationMutDiag);
                    }
                };
            }
        }
    }
}
