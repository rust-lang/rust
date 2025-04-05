use clippy_utils::diagnostics::{span_lint_hir, span_lint_hir_and_then};
use clippy_utils::source::SpanRangeExt;
use clippy_utils::ty::has_drop;
use clippy_utils::{
    in_automatically_derived, is_inside_always_const_context, is_lint_allowed, path_to_local, peel_blocks,
};
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{
    BinOpKind, BlockCheckMode, Expr, ExprKind, HirId, HirIdMap, ItemKind, LocalSource, Node, PatKind, Stmt, StmtKind,
    StructTailExpr, UnsafeSource, is_range_literal,
};
use rustc_infer::infer::TyCtxtInferExt as _;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use std::ops::Deref;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for statements which have no effect.
    ///
    /// ### Why is this bad?
    /// Unlike dead code, these statements are actually
    /// executed. However, as they have no effect, all they do is make the code less
    /// readable.
    ///
    /// ### Example
    /// ```no_run
    /// 0;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NO_EFFECT,
    complexity,
    "statements with no effect"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for binding to underscore prefixed variable without side-effects.
    ///
    /// ### Why is this bad?
    /// Unlike dead code, these bindings are actually
    /// executed. However, as they have no effect and shouldn't be used further on, all they
    /// do is make the code less readable.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let _i_serve_no_purpose = 1;
    /// ```
    #[clippy::version = "1.58.0"]
    pub NO_EFFECT_UNDERSCORE_BINDING,
    pedantic,
    "binding to `_` prefixed variable with no side-effect"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for expression statements that can be reduced to a
    /// sub-expression.
    ///
    /// ### Why is this bad?
    /// Expressions by themselves often have no side-effects.
    /// Having such expressions reduces readability.
    ///
    /// ### Example
    /// ```rust,ignore
    /// compute_array()[0];
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub UNNECESSARY_OPERATION,
    complexity,
    "outer expressions with no effect"
}

#[derive(Default)]
pub struct NoEffect {
    underscore_bindings: HirIdMap<Span>,
    local_bindings: Vec<Vec<HirId>>,
}

impl_lint_pass!(NoEffect => [NO_EFFECT, UNNECESSARY_OPERATION, NO_EFFECT_UNDERSCORE_BINDING]);

impl<'tcx> LateLintPass<'tcx> for NoEffect {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if self.check_no_effect(cx, stmt) {
            return;
        }
        check_unnecessary_operation(cx, stmt);
    }

    fn check_block(&mut self, _: &LateContext<'tcx>, _: &'tcx rustc_hir::Block<'tcx>) {
        self.local_bindings.push(Vec::default());
    }

    fn check_block_post(&mut self, cx: &LateContext<'tcx>, _: &'tcx rustc_hir::Block<'tcx>) {
        for hir_id in self.local_bindings.pop().unwrap() {
            if let Some(span) = self.underscore_bindings.swap_remove(&hir_id) {
                span_lint_hir(
                    cx,
                    NO_EFFECT_UNDERSCORE_BINDING,
                    hir_id,
                    span,
                    "binding to `_` prefixed variable with no side-effect",
                );
            }
        }
    }

    fn check_expr(&mut self, _: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let Some(def_id) = path_to_local(expr) {
            self.underscore_bindings.swap_remove(&def_id);
        }
    }
}

impl NoEffect {
    fn check_no_effect(&mut self, cx: &LateContext<'_>, stmt: &Stmt<'_>) -> bool {
        if let StmtKind::Semi(expr) = stmt.kind {
            // Covered by rustc `path_statements` lint
            if matches!(expr.kind, ExprKind::Path(_)) {
                return true;
            }

            if expr.span.from_expansion() {
                return false;
            }
            let expr = peel_blocks(expr);

            if is_operator_overridden(cx, expr) {
                // Return `true`, to prevent `check_unnecessary_operation` from
                // linting on this statement as well.
                return true;
            }
            if has_no_effect(cx, expr) {
                span_lint_hir_and_then(
                    cx,
                    NO_EFFECT,
                    expr.hir_id,
                    stmt.span,
                    "statement with no effect",
                    |diag| {
                        for parent in cx.tcx.hir_parent_iter(stmt.hir_id) {
                            if let Node::Item(item) = parent.1
                                && let ItemKind::Fn { .. } = item.kind
                                && let Node::Block(block) = cx.tcx.parent_hir_node(stmt.hir_id)
                                && let [.., final_stmt] = block.stmts
                                && final_stmt.hir_id == stmt.hir_id
                            {
                                let expr_ty = cx.typeck_results().expr_ty(expr);
                                let mut ret_ty = cx
                                    .tcx
                                    .fn_sig(item.owner_id)
                                    .instantiate_identity()
                                    .output()
                                    .skip_binder();

                                // Remove `impl Future<Output = T>` to get `T`
                                if cx.tcx.ty_is_opaque_future(ret_ty)
                                    && let Some(true_ret_ty) = cx
                                        .tcx
                                        .infer_ctxt()
                                        .build(cx.typing_mode())
                                        .err_ctxt()
                                        .get_impl_future_output_ty(ret_ty)
                                {
                                    ret_ty = true_ret_ty;
                                }

                                if !ret_ty.is_unit() && ret_ty == expr_ty {
                                    diag.span_suggestion(
                                        stmt.span.shrink_to_lo(),
                                        "did you mean to return it?",
                                        "return ",
                                        Applicability::MaybeIncorrect,
                                    );
                                }
                            }
                        }
                    },
                );
                return true;
            }
        } else if let StmtKind::Let(local) = stmt.kind
            && !is_lint_allowed(cx, NO_EFFECT_UNDERSCORE_BINDING, local.hir_id)
            && !matches!(local.source, LocalSource::AsyncFn)
            && let Some(init) = local.init
            && local.els.is_none()
            && !local.pat.span.from_expansion()
            && has_no_effect(cx, init)
            && let PatKind::Binding(_, hir_id, ident, _) = local.pat.kind
            && ident.name.to_ident_string().starts_with('_')
            && !in_automatically_derived(cx.tcx, local.hir_id)
        {
            if let Some(l) = self.local_bindings.last_mut() {
                l.push(hir_id);
                self.underscore_bindings.insert(hir_id, ident.span);
            }
            return true;
        }
        false
    }
}

fn is_operator_overridden(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    // It's very hard or impossible to check whether overridden operator have side-effect this lint.
    // So, this function assume user-defined operator is overridden with an side-effect.
    // The definition of user-defined structure here is ADT-type,
    // Althrough this will weaken the ability of this lint, less error lint-fix happen.
    match expr.kind {
        ExprKind::Binary(..) | ExprKind::Unary(..) => {
            // No need to check type of `lhs` and `rhs`
            // because if the operator is overridden, at least one operand is ADT type

            // reference: rust/compiler/rustc_middle/src/ty/typeck_results.rs: `is_method_call`.
            // use this function to check whether operator is overridden in `ExprKind::{Binary, Unary}`.
            cx.typeck_results().is_method_call(expr)
        },
        _ => false,
    }
}

fn has_no_effect(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    match expr.kind {
        ExprKind::Lit(..) | ExprKind::Closure { .. } => true,
        ExprKind::Path(..) => !has_drop(cx, cx.typeck_results().expr_ty(expr)),
        ExprKind::Index(a, b, _) | ExprKind::Binary(_, a, b) => has_no_effect(cx, a) && has_no_effect(cx, b),
        ExprKind::Array(v) | ExprKind::Tup(v) => v.iter().all(|val| has_no_effect(cx, val)),
        ExprKind::Repeat(inner, _)
        | ExprKind::Cast(inner, _)
        | ExprKind::Type(inner, _)
        | ExprKind::Unary(_, inner)
        | ExprKind::Field(inner, _)
        | ExprKind::AddrOf(_, _, inner) => has_no_effect(cx, inner),
        ExprKind::Struct(_, fields, ref base) => {
            !has_drop(cx, cx.typeck_results().expr_ty(expr))
                && fields.iter().all(|field| has_no_effect(cx, field.expr))
                && match &base {
                    StructTailExpr::None | StructTailExpr::DefaultFields(_) => true,
                    StructTailExpr::Base(base) => has_no_effect(cx, base),
                }
        },
        ExprKind::Call(callee, args) => {
            if let ExprKind::Path(ref qpath) = callee.kind {
                if cx.typeck_results().type_dependent_def(expr.hir_id).is_some() {
                    // type-dependent function call like `impl FnOnce for X`
                    return false;
                }
                let def_matched = matches!(
                    cx.qpath_res(qpath, callee.hir_id),
                    Res::Def(DefKind::Struct | DefKind::Variant | DefKind::Ctor(..), ..)
                );
                if def_matched || is_range_literal(expr) {
                    !has_drop(cx, cx.typeck_results().expr_ty(expr)) && args.iter().all(|arg| has_no_effect(cx, arg))
                } else {
                    false
                }
            } else {
                false
            }
        },
        _ => false,
    }
}

fn check_unnecessary_operation(cx: &LateContext<'_>, stmt: &Stmt<'_>) {
    if let StmtKind::Semi(expr) = stmt.kind
        && !stmt.span.in_external_macro(cx.sess().source_map())
        && let ctxt = stmt.span.ctxt()
        && expr.span.ctxt() == ctxt
        && let Some(reduced) = reduce_expression(cx, expr)
        && reduced.iter().all(|e| e.span.ctxt() == ctxt)
    {
        if let ExprKind::Index(..) = &expr.kind {
            if !is_inside_always_const_context(cx.tcx, expr.hir_id)
                && let [arr, func] = &*reduced
                && let Some(arr) = arr.span.get_source_text(cx)
                && let Some(func) = func.span.get_source_text(cx)
            {
                span_lint_hir_and_then(
                    cx,
                    UNNECESSARY_OPERATION,
                    expr.hir_id,
                    stmt.span,
                    "unnecessary operation",
                    |diag| {
                        diag.span_suggestion(
                            stmt.span,
                            "statement can be written as",
                            format!("assert!({arr}.len() > {func});"),
                            Applicability::MaybeIncorrect,
                        );
                    },
                );
            }
        } else {
            let mut snippet = String::new();
            for e in reduced {
                if let Some(snip) = e.span.get_source_text(cx) {
                    snippet.push_str(&snip);
                    snippet.push(';');
                } else {
                    return;
                }
            }
            span_lint_hir_and_then(
                cx,
                UNNECESSARY_OPERATION,
                expr.hir_id,
                stmt.span,
                "unnecessary operation",
                |diag| {
                    diag.span_suggestion(
                        stmt.span,
                        "statement can be reduced to",
                        snippet,
                        Applicability::MachineApplicable,
                    );
                },
            );
        }
    }
}

fn reduce_expression<'a>(cx: &LateContext<'_>, expr: &'a Expr<'a>) -> Option<Vec<&'a Expr<'a>>> {
    if expr.span.from_expansion() {
        return None;
    }
    match expr.kind {
        ExprKind::Index(a, b, _) => Some(vec![a, b]),
        ExprKind::Binary(ref binop, a, b) if binop.node != BinOpKind::And && binop.node != BinOpKind::Or => {
            Some(vec![a, b])
        },
        ExprKind::Array(v) | ExprKind::Tup(v) => Some(v.iter().collect()),
        ExprKind::Repeat(inner, _)
        | ExprKind::Cast(inner, _)
        | ExprKind::Type(inner, _)
        | ExprKind::Unary(_, inner)
        | ExprKind::Field(inner, _)
        | ExprKind::AddrOf(_, _, inner) => reduce_expression(cx, inner).or_else(|| Some(vec![inner])),
        ExprKind::Struct(_, fields, ref base) => {
            if has_drop(cx, cx.typeck_results().expr_ty(expr)) {
                None
            } else {
                let base = match base {
                    StructTailExpr::Base(base) => Some(base),
                    StructTailExpr::None | StructTailExpr::DefaultFields(_) => None,
                };
                Some(fields.iter().map(|f| &f.expr).chain(base).map(Deref::deref).collect())
            }
        },
        ExprKind::Call(callee, args) => {
            if let ExprKind::Path(ref qpath) = callee.kind {
                if cx.typeck_results().type_dependent_def(expr.hir_id).is_some() {
                    // type-dependent function call like `impl FnOnce for X`
                    return None;
                }
                let res = cx.qpath_res(qpath, callee.hir_id);
                match res {
                    Res::Def(DefKind::Struct | DefKind::Variant | DefKind::Ctor(..), ..)
                        if !has_drop(cx, cx.typeck_results().expr_ty(expr)) =>
                    {
                        Some(args.iter().collect())
                    },
                    _ => None,
                }
            } else {
                None
            }
        },
        ExprKind::Block(block, _) => {
            if block.stmts.is_empty() && !block.targeted_by_break {
                block.expr.as_ref().and_then(|e| {
                    match block.rules {
                        BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided) => None,
                        BlockCheckMode::DefaultBlock => Some(vec![&**e]),
                        // in case of compiler-inserted signaling blocks
                        BlockCheckMode::UnsafeBlock(_) => reduce_expression(cx, e),
                    }
                })
            } else {
                None
            }
        },
        _ => None,
    }
}
