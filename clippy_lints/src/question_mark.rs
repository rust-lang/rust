use crate::manual_let_else::MANUAL_LET_ELSE;
use crate::question_mark_used::QUESTION_MARK_USED;
use clippy_config::msrvs::Msrv;
use clippy_config::types::MatchLintBehaviour;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{
    eq_expr_value, higher, in_constant, is_else_clause, is_lint_allowed, is_path_lang_item, is_res_lang_ctor,
    pat_and_expr_can_be_question_mark, path_to_local, path_to_local_id, peel_blocks, peel_blocks_with_stmt,
    span_contains_comment,
};
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::LangItem::{self, OptionNone, OptionSome, ResultErr, ResultOk};
use rustc_hir::{
    BindingAnnotation, Block, ByRef, Expr, ExprKind, Local, Node, PatKind, PathSegment, QPath, Stmt, StmtKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Ty;
use rustc_session::impl_lint_pass;
use rustc_span::sym;
use rustc_span::symbol::Symbol;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for expressions that could be replaced by the question mark operator.
    ///
    /// ### Why is this bad?
    /// Question mark usage is more idiomatic.
    ///
    /// ### Example
    /// ```ignore
    /// if option.is_none() {
    ///     return None;
    /// }
    /// ```
    ///
    /// Could be written:
    ///
    /// ```ignore
    /// option?;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub QUESTION_MARK,
    style,
    "checks for expressions that could be replaced by the question mark operator"
}

pub struct QuestionMark {
    pub(crate) msrv: Msrv,
    pub(crate) matches_behaviour: MatchLintBehaviour,
    /// Keeps track of how many try blocks we are in at any point during linting.
    /// This allows us to answer the question "are we inside of a try block"
    /// very quickly, without having to walk up the parent chain, by simply checking
    /// if it is greater than zero.
    /// As for why we need this in the first place: <https://github.com/rust-lang/rust-clippy/issues/8628>
    try_block_depth_stack: Vec<u32>,
}

impl_lint_pass!(QuestionMark => [QUESTION_MARK, MANUAL_LET_ELSE]);

impl QuestionMark {
    #[must_use]
    pub fn new(msrv: Msrv, matches_behaviour: MatchLintBehaviour) -> Self {
        Self {
            msrv,
            matches_behaviour,
            try_block_depth_stack: Vec::new(),
        }
    }
}

enum IfBlockType<'hir> {
    /// An `if x.is_xxx() { a } else { b } ` expression.
    ///
    /// Contains: `caller (x), caller_type, call_sym (is_xxx), if_then (a), if_else (b)`
    IfIs(&'hir Expr<'hir>, Ty<'hir>, Symbol, &'hir Expr<'hir>),
    /// An `if let Xxx(a) = b { c } else { d }` expression.
    ///
    /// Contains: `let_pat_qpath (Xxx), let_pat_type, let_pat_sym (a), let_expr (b), if_then (c),
    /// if_else (d)`
    IfLet(
        Res,
        Ty<'hir>,
        Symbol,
        &'hir Expr<'hir>,
        &'hir Expr<'hir>,
        Option<&'hir Expr<'hir>>,
    ),
}

fn find_let_else_ret_expression<'hir>(block: &'hir Block<'hir>) -> Option<&'hir Expr<'hir>> {
    if let Block {
        stmts: &[],
        expr: Some(els),
        ..
    } = block
    {
        Some(els)
    } else if let [stmt] = block.stmts
        && let StmtKind::Semi(expr) = stmt.kind
        && let ExprKind::Ret(..) = expr.kind
    {
        Some(expr)
    } else {
        None
    }
}

fn check_let_some_else_return_none(cx: &LateContext<'_>, stmt: &Stmt<'_>) {
    if let StmtKind::Let(Local {
        pat,
        init: Some(init_expr),
        els: Some(els),
        ..
    }) = stmt.kind
        && let Some(ret) = find_let_else_ret_expression(els)
        && let Some(inner_pat) = pat_and_expr_can_be_question_mark(cx, pat, ret)
        && !span_contains_comment(cx.tcx.sess.source_map(), els.span)
    {
        let mut applicability = Applicability::MaybeIncorrect;
        let init_expr_str = snippet_with_applicability(cx, init_expr.span, "..", &mut applicability);
        let receiver_str = snippet_with_applicability(cx, inner_pat.span, "..", &mut applicability);
        let sugg = format!("let {receiver_str} = {init_expr_str}?;",);
        span_lint_and_sugg(
            cx,
            QUESTION_MARK,
            stmt.span,
            "this `let...else` may be rewritten with the `?` operator",
            "replace it with",
            sugg,
            applicability,
        );
    }
}

fn is_early_return(smbl: Symbol, cx: &LateContext<'_>, if_block: &IfBlockType<'_>) -> bool {
    match *if_block {
        IfBlockType::IfIs(caller, caller_ty, call_sym, if_then) => {
            // If the block could be identified as `if x.is_none()/is_err()`,
            // we then only need to check the if_then return to see if it is none/err.
            is_type_diagnostic_item(cx, caller_ty, smbl)
                && expr_return_none_or_err(smbl, cx, if_then, caller, None)
                && match smbl {
                    sym::Option => call_sym == sym!(is_none),
                    sym::Result => call_sym == sym!(is_err),
                    _ => false,
                }
        },
        IfBlockType::IfLet(res, let_expr_ty, let_pat_sym, let_expr, if_then, if_else) => {
            is_type_diagnostic_item(cx, let_expr_ty, smbl)
                && match smbl {
                    sym::Option => {
                        // We only need to check `if let Some(x) = option` not `if let None = option`,
                        // because the later one will be suggested as `if option.is_none()` thus causing conflict.
                        is_res_lang_ctor(cx, res, OptionSome)
                            && if_else.is_some()
                            && expr_return_none_or_err(smbl, cx, if_else.unwrap(), let_expr, None)
                    },
                    sym::Result => {
                        (is_res_lang_ctor(cx, res, ResultOk)
                            && if_else.is_some()
                            && expr_return_none_or_err(smbl, cx, if_else.unwrap(), let_expr, Some(let_pat_sym)))
                            || is_res_lang_ctor(cx, res, ResultErr)
                                && expr_return_none_or_err(smbl, cx, if_then, let_expr, Some(let_pat_sym))
                                && if_else.is_none()
                    },
                    _ => false,
                }
        },
    }
}

fn expr_return_none_or_err(
    smbl: Symbol,
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    cond_expr: &Expr<'_>,
    err_sym: Option<Symbol>,
) -> bool {
    match peel_blocks_with_stmt(expr).kind {
        ExprKind::Ret(Some(ret_expr)) => expr_return_none_or_err(smbl, cx, ret_expr, cond_expr, err_sym),
        ExprKind::Path(ref qpath) => match smbl {
            sym::Option => is_res_lang_ctor(cx, cx.qpath_res(qpath, expr.hir_id), OptionNone),
            sym::Result => path_to_local(expr).is_some() && path_to_local(expr) == path_to_local(cond_expr),
            _ => false,
        },
        ExprKind::Call(call_expr, args_expr) => {
            if smbl == sym::Result
                && let ExprKind::Path(QPath::Resolved(_, path)) = &call_expr.kind
                && let Some(segment) = path.segments.first()
                && let Some(err_sym) = err_sym
                && let Some(arg) = args_expr.first()
                && let ExprKind::Path(QPath::Resolved(_, arg_path)) = &arg.kind
                && let Some(PathSegment { ident, .. }) = arg_path.segments.first()
            {
                return segment.ident.name == sym::Err && err_sym == ident.name;
            }
            false
        },
        _ => false,
    }
}

/// Checks if the given expression on the given context matches the following structure:
///
/// ```ignore
/// if option.is_none() {
///    return None;
/// }
/// ```
///
/// ```ignore
/// if result.is_err() {
///     return result;
/// }
/// ```
///
/// If it matches, it will suggest to use the question mark operator instead
fn check_is_none_or_err_and_early_return<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
    if let Some(higher::If { cond, then, r#else }) = higher::If::hir(expr)
        && !is_else_clause(cx.tcx, expr)
        && let ExprKind::MethodCall(segment, caller, ..) = &cond.kind
        && let caller_ty = cx.typeck_results().expr_ty(caller)
        && let if_block = IfBlockType::IfIs(caller, caller_ty, segment.ident.name, then)
        && (is_early_return(sym::Option, cx, &if_block) || is_early_return(sym::Result, cx, &if_block))
    {
        let mut applicability = Applicability::MachineApplicable;
        let receiver_str = snippet_with_applicability(cx, caller.span, "..", &mut applicability);
        let by_ref = !caller_ty.is_copy_modulo_regions(cx.tcx, cx.param_env)
            && !matches!(caller.kind, ExprKind::Call(..) | ExprKind::MethodCall(..));
        let sugg = if let Some(else_inner) = r#else {
            if eq_expr_value(cx, caller, peel_blocks(else_inner)) {
                format!("Some({receiver_str}?)")
            } else {
                return;
            }
        } else {
            format!("{receiver_str}{}?;", if by_ref { ".as_ref()" } else { "" })
        };

        span_lint_and_sugg(
            cx,
            QUESTION_MARK,
            expr.span,
            "this block may be rewritten with the `?` operator",
            "replace it with",
            sugg,
            applicability,
        );
    }
}

fn check_if_let_some_or_err_and_early_return<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
    if let Some(higher::IfLet {
        let_pat,
        let_expr,
        if_then,
        if_else,
        ..
    }) = higher::IfLet::hir(cx, expr)
        && !is_else_clause(cx.tcx, expr)
        && let PatKind::TupleStruct(ref path1, [field], ddpos) = let_pat.kind
        && ddpos.as_opt_usize().is_none()
        && let PatKind::Binding(BindingAnnotation(by_ref, _), bind_id, ident, None) = field.kind
        && let caller_ty = cx.typeck_results().expr_ty(let_expr)
        && let if_block = IfBlockType::IfLet(
            cx.qpath_res(path1, let_pat.hir_id),
            caller_ty,
            ident.name,
            let_expr,
            if_then,
            if_else,
        )
        && ((is_early_return(sym::Option, cx, &if_block) && path_to_local_id(peel_blocks(if_then), bind_id))
            || is_early_return(sym::Result, cx, &if_block))
        && if_else
            .map(|e| eq_expr_value(cx, let_expr, peel_blocks(e)))
            .filter(|e| *e)
            .is_none()
    {
        let mut applicability = Applicability::MachineApplicable;
        let receiver_str = snippet_with_applicability(cx, let_expr.span, "..", &mut applicability);
        let requires_semi = matches!(cx.tcx.parent_hir_node(expr.hir_id), Node::Stmt(_));
        let sugg = format!(
            "{receiver_str}{}?{}",
            if by_ref == ByRef::Yes { ".as_ref()" } else { "" },
            if requires_semi { ";" } else { "" }
        );
        span_lint_and_sugg(
            cx,
            QUESTION_MARK,
            expr.span,
            "this block may be rewritten with the `?` operator",
            "replace it with",
            sugg,
            applicability,
        );
    }
}

impl QuestionMark {
    fn inside_try_block(&self) -> bool {
        self.try_block_depth_stack.last() > Some(&0)
    }
}

fn is_try_block(cx: &LateContext<'_>, bl: &rustc_hir::Block<'_>) -> bool {
    if let Some(expr) = bl.expr
        && let rustc_hir::ExprKind::Call(callee, _) = expr.kind
    {
        is_path_lang_item(cx, callee, LangItem::TryTraitFromOutput)
    } else {
        false
    }
}

impl<'tcx> LateLintPass<'tcx> for QuestionMark {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if !is_lint_allowed(cx, QUESTION_MARK_USED, stmt.hir_id) {
            return;
        }

        if !self.inside_try_block() && !in_constant(cx, stmt.hir_id) {
            check_let_some_else_return_none(cx, stmt);
        }
        self.check_manual_let_else(cx, stmt);
    }
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !self.inside_try_block()
            && !in_constant(cx, expr.hir_id)
            && is_lint_allowed(cx, QUESTION_MARK_USED, expr.hir_id)
        {
            check_is_none_or_err_and_early_return(cx, expr);
            check_if_let_some_or_err_and_early_return(cx, expr);
        }
    }

    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx rustc_hir::Block<'tcx>) {
        if is_try_block(cx, block) {
            *self
                .try_block_depth_stack
                .last_mut()
                .expect("blocks are always part of bodies and must have a depth") += 1;
        }
    }

    fn check_body(&mut self, _: &LateContext<'tcx>, _: &'tcx rustc_hir::Body<'tcx>) {
        self.try_block_depth_stack.push(0);
    }

    fn check_body_post(&mut self, _: &LateContext<'tcx>, _: &'tcx rustc_hir::Body<'tcx>) {
        self.try_block_depth_stack.pop();
    }

    fn check_block_post(&mut self, cx: &LateContext<'tcx>, block: &'tcx rustc_hir::Block<'tcx>) {
        if is_try_block(cx, block) {
            *self
                .try_block_depth_stack
                .last_mut()
                .expect("blocks are always part of bodies and must have a depth") -= 1;
        }
    }
    extract_msrv_attr!(LateContext);
}
