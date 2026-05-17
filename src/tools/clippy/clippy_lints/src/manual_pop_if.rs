use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::res::MaybeDef;
use clippy_utils::source::snippet_with_context;
use clippy_utils::visitors::{for_each_expr_without_closures, is_local_used};
use clippy_utils::{eq_expr_value, is_else_clause, is_lang_item_or_ctor, span_contains_non_whitespace, sym};
use rustc_ast::LitKind;
use rustc_errors::{Applicability, MultiSpan};
use rustc_hir::{BlockCheckMode, Expr, ExprKind, LangItem, PatKind, StmtKind, UnsafeSource};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TyCtxt;
use rustc_session::impl_lint_pass;
use rustc_span::{BytePos, Span, Symbol};
use std::fmt;
use std::ops::ControlFlow;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for code to be replaced by `pop_if` methods.
    ///
    /// ### Why is this bad?
    /// Using `pop_if` is more concise and idiomatic.
    ///
    /// ### Known issues
    /// When the popped value is assigned or used in an expression,
    /// or when the `if` condition is part of an `else if` branch, the
    /// lint will trigger but will not provide an automatic suggestion.
    ///
    /// ### Examples
    /// ```no_run
    /// # use std::collections::VecDeque;
    /// # let mut vec = vec![1, 2, 3, 4, 5];
    /// # let mut deque: VecDeque<i32> = VecDeque::from([1, 2, 3, 4, 5]);
    /// if vec.last().is_some_and(|x| *x > 5) {
    ///     vec.pop().unwrap();
    /// }
    /// if deque.back().is_some_and(|x| *x > 5) {
    ///     deque.pop_back().unwrap();
    /// }
    /// if deque.front().is_some_and(|x| *x > 5) {
    ///     deque.pop_front().unwrap();
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// # use std::collections::VecDeque;
    /// # let mut vec = vec![1, 2, 3, 4, 5];
    /// # let mut deque: VecDeque<i32> = VecDeque::from([1, 2, 3, 4, 5]);
    /// vec.pop_if(|x| *x > 5);
    /// deque.pop_back_if(|x| *x > 5);
    /// deque.pop_front_if(|x| *x > 5);
    /// ```
    #[clippy::version = "1.95.0"]
    pub MANUAL_POP_IF,
    complexity,
    "manual implementation of `pop_if` methods"
}

impl_lint_pass!(ManualPopIf => [MANUAL_POP_IF]);

pub struct ManualPopIf {
    msrv: Msrv,
    binary_heap_pop_if_feature_enabled: bool,
}

impl ManualPopIf {
    pub fn new(tcx: TyCtxt<'_>, conf: &'static Conf) -> Self {
        Self {
            msrv: conf.msrv,
            binary_heap_pop_if_feature_enabled: tcx.features().enabled(sym::binary_heap_pop_if),
        }
    }

    fn msrv_compatible(&self, cx: &LateContext<'_>, kind: ManualPopIfKind) -> bool {
        match kind {
            ManualPopIfKind::Vec => self.msrv.meets(cx, msrvs::VEC_POP_IF),
            ManualPopIfKind::VecDequeBack => self.msrv.meets(cx, msrvs::VEC_DEQUE_POP_BACK_IF),
            ManualPopIfKind::VecDequeFront => self.msrv.meets(cx, msrvs::VEC_DEQUE_POP_FRONT_IF),
            ManualPopIfKind::BinaryHeap => self.binary_heap_pop_if_feature_enabled,
        }
    }
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManualPopIfKind {
    Vec,
    VecDequeBack,
    VecDequeFront,
    BinaryHeap,
}

impl ManualPopIfKind {
    fn peek_method(self) -> Symbol {
        match self {
            ManualPopIfKind::Vec => sym::last,
            ManualPopIfKind::VecDequeBack => sym::back,
            ManualPopIfKind::VecDequeFront => sym::front,
            ManualPopIfKind::BinaryHeap => sym::peek,
        }
    }

    fn pop_method(self) -> Symbol {
        match self {
            ManualPopIfKind::Vec | ManualPopIfKind::BinaryHeap => sym::pop,
            ManualPopIfKind::VecDequeBack => sym::pop_back,
            ManualPopIfKind::VecDequeFront => sym::pop_front,
        }
    }

    fn pop_if_method(self) -> Symbol {
        match self {
            ManualPopIfKind::Vec | ManualPopIfKind::BinaryHeap => sym::pop_if,
            ManualPopIfKind::VecDequeBack => sym::pop_back_if,
            ManualPopIfKind::VecDequeFront => sym::pop_front_if,
        }
    }

    fn is_diag_item(self, cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
        let ty = cx.typeck_results().expr_ty(expr).peel_refs();
        match self {
            ManualPopIfKind::Vec => ty.is_diag_item(cx, sym::Vec),
            ManualPopIfKind::VecDequeBack | ManualPopIfKind::VecDequeFront => ty.is_diag_item(cx, sym::VecDeque),
            ManualPopIfKind::BinaryHeap => ty.is_diag_item(cx, sym::BinaryHeap),
        }
    }
}

impl fmt::Display for ManualPopIfKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ManualPopIfKind::Vec => write!(f, "`Vec::pop_if`"),
            ManualPopIfKind::VecDequeBack => write!(f, "`VecDeque::pop_back_if`"),
            ManualPopIfKind::VecDequeFront => write!(f, "`VecDeque::pop_front_if`"),
            ManualPopIfKind::BinaryHeap => write!(f, "`BinaryHeap::pop_if`"),
        }
    }
}

struct ManualPopIfPattern<'tcx> {
    kind: ManualPopIfKind,

    /// The collection (`vec` in `vec.last()`)
    collection_expr: &'tcx Expr<'tcx>,

    /// The closure (`*x > 5` in `|x| *x > 5`)
    predicate: &'tcx Expr<'tcx>,

    /// Parameter name for the closure (`x` in `|x| *x > 5`)
    param_name: Symbol,

    /// Span of the if expression (including the `if` keyword)
    if_span: Span,

    /// Span of the:
    /// - check call (`vec.last().is_some_and(|x| *x > 5)`)
    /// - pop+unwrap call (`vec.pop().unwrap()`)
    spans: MultiSpan,

    /// Whether we are able to provide a suggestion
    suggestable: bool,
}

impl ManualPopIfPattern<'_> {
    fn emit_lint(self, cx: &LateContext<'_>) {
        let mut app = Applicability::MachineApplicable;
        let ctxt = self.if_span.ctxt();
        let collection_snippet = snippet_with_context(cx, self.collection_expr.span, ctxt, "..", &mut app).0;
        let predicate_snippet = snippet_with_context(cx, self.predicate.span, ctxt, "..", &mut app).0;
        let param_name = self.param_name;
        let pop_if_method = self.kind.pop_if_method();

        span_lint_and_then(
            cx,
            MANUAL_POP_IF,
            self.spans,
            format!("manual implementation of {}", self.kind),
            |diag| {
                let sugg = format!("{collection_snippet}.{pop_if_method}(|{param_name}| {predicate_snippet});");
                if self.suggestable {
                    diag.span_suggestion_verbose(self.if_span, "try", sugg, app);
                } else {
                    diag.help(format!("try refactoring the code using `{sugg}`"));
                }
            },
        );
    }
}

/// Checks for the pattern:
/// ```ignore
/// if vec.last().is_some_and(|x| *x > 5) {
///     vec.pop().unwrap();
/// }
/// ```
fn check_is_some_and_pattern<'tcx>(
    cx: &LateContext<'tcx>,
    cond: &'tcx Expr<'_>,
    then_block: &'tcx Expr<'_>,
    if_expr_span: Span,
    kind: ManualPopIfKind,
) -> Option<ManualPopIfPattern<'tcx>> {
    let peek_method = kind.peek_method();
    let pop_method = kind.pop_method();

    if let ExprKind::MethodCall(path, receiver, [closure_arg], _) = cond.kind
        && path.ident.name == sym::is_some_and
        && let ExprKind::MethodCall(check_path, collection_expr, [], _) = receiver.kind
        && check_path.ident.name == peek_method
        && kind.is_diag_item(cx, collection_expr)
        && let ExprKind::Closure(closure) = closure_arg.kind
        && let body = cx.tcx.hir_body(closure.body)
        && let Some((pop_collection, pop_span, suggestable)) = check_pop_unwrap(cx, then_block, pop_method)
        && eq_expr_value(cx, if_expr_span.ctxt(), collection_expr, pop_collection)
        && let Some(param) = body.params.first()
        && let Some(ident) = param.pat.simple_ident()
    {
        return Some(ManualPopIfPattern {
            kind,
            collection_expr,
            predicate: body.value,
            param_name: ident.name,
            if_span: if_expr_span,
            spans: MultiSpan::from(vec![if_expr_span.with_hi(cond.span.hi()), pop_span]),
            suggestable,
        });
    }

    None
}

/// Checks for the pattern:
/// ```ignore
/// if let Some(x) = vec.last() {
///     if *x > 5 {
///         vec.pop().unwrap();
///     }
/// }
/// ```
fn check_if_let_pattern<'tcx>(
    cx: &LateContext<'tcx>,
    cond: &'tcx Expr<'_>,
    then_block: &'tcx Expr<'_>,
    if_expr_span: Span,
    kind: ManualPopIfKind,
) -> Option<ManualPopIfPattern<'tcx>> {
    let peek_method = kind.peek_method();
    let pop_method = kind.pop_method();

    if let ExprKind::Let(let_expr) = cond.kind
        && let PatKind::TupleStruct(qpath, [binding_pat], _) = let_expr.pat.kind
    {
        let res = cx.qpath_res(&qpath, let_expr.pat.hir_id);

        if let Some(def_id) = res.opt_def_id()
            && is_lang_item_or_ctor(cx, def_id, LangItem::OptionSome)
            && let PatKind::Binding(_, binding_id, binding_name, _) = binding_pat.kind
            && let ExprKind::MethodCall(path, collection_expr, [], _) = let_expr.init.kind
            && path.ident.name == peek_method
            && kind.is_diag_item(cx, collection_expr)
            && let ExprKind::Block(block, _) = then_block.kind
        {
            // The inner if can be either a statement or a block expression
            let inner_if = match (block.stmts, block.expr) {
                ([stmt], _) => match stmt.kind {
                    StmtKind::Expr(expr) | StmtKind::Semi(expr) => expr,
                    _ => return None,
                },
                ([], Some(expr)) => expr,
                _ => return None,
            };

            if let ExprKind::If(inner_cond, inner_then, None) = inner_if.kind
                && is_local_used(cx, inner_cond, binding_id)
                && let Some((pop_collection, pop_span, suggestable)) = check_pop_unwrap(cx, inner_then, pop_method)
                && eq_expr_value(cx, if_expr_span.ctxt(), collection_expr, pop_collection)
            {
                return Some(ManualPopIfPattern {
                    kind,
                    collection_expr,
                    predicate: inner_cond,
                    param_name: binding_name.name,
                    if_span: if_expr_span,
                    spans: MultiSpan::from(vec![
                        if_expr_span.with_hi(cond.span.hi()),
                        inner_if.span.with_hi(inner_cond.span.hi()),
                        pop_span,
                    ]),
                    suggestable,
                });
            }
        }
    }

    None
}

/// Checks for the pattern:
/// ```ignore
/// if let Some(x) = vec.last() && *x > 5 {
///     vec.pop().unwrap();
/// }
/// ```
fn check_let_chain_pattern<'tcx>(
    cx: &LateContext<'tcx>,
    cond: &'tcx Expr<'_>,
    then_block: &'tcx Expr<'_>,
    if_expr_span: Span,
    kind: ManualPopIfKind,
) -> Option<ManualPopIfPattern<'tcx>> {
    let peek_method = kind.peek_method();
    let pop_method = kind.pop_method();

    if let ExprKind::Binary(op, left, right) = cond.kind
        && op.node == rustc_ast::BinOpKind::And
        && let ExprKind::Let(let_expr) = left.kind
        && let PatKind::TupleStruct(qpath, [binding_pat], _) = let_expr.pat.kind
    {
        let res = cx.qpath_res(&qpath, let_expr.pat.hir_id);

        if let Some(def_id) = res.opt_def_id()
            && is_lang_item_or_ctor(cx, def_id, LangItem::OptionSome)
            && let PatKind::Binding(_, binding_id, binding_name, _) = binding_pat.kind
            && let ExprKind::MethodCall(path, collection_expr, [], _) = let_expr.init.kind
            && path.ident.name == peek_method
            && kind.is_diag_item(cx, collection_expr)
            && is_local_used(cx, right, binding_id)
            && let Some((pop_collection, pop_span, suggestable)) = check_pop_unwrap(cx, then_block, pop_method)
            && eq_expr_value(cx, if_expr_span.ctxt(), collection_expr, pop_collection)
        {
            return Some(ManualPopIfPattern {
                kind,
                collection_expr,
                predicate: right,
                param_name: binding_name.name,
                if_span: if_expr_span,
                spans: MultiSpan::from(vec![if_expr_span.with_hi(cond.span.hi()), pop_span]),
                suggestable,
            });
        }
    }

    None
}

/// Checks for the pattern:
/// ```ignore
/// if vec.last().map(|x| *x > 5).unwrap_or(false) {
///     vec.pop().unwrap();
/// }
/// ```
fn check_map_unwrap_or_pattern<'tcx>(
    cx: &LateContext<'tcx>,
    cond: &'tcx Expr<'_>,
    then_block: &'tcx Expr<'_>,
    if_expr_span: Span,
    kind: ManualPopIfKind,
) -> Option<ManualPopIfPattern<'tcx>> {
    let peek_method = kind.peek_method();
    let pop_method = kind.pop_method();

    if let ExprKind::MethodCall(unwrap_path, receiver, [default_arg], _) = cond.kind
        && unwrap_path.ident.name == sym::unwrap_or
        && matches!(default_arg.kind, ExprKind::Lit(lit) if matches!(lit.node, LitKind::Bool(false)))
        && let ExprKind::MethodCall(map_path, map_receiver, [closure_arg], _) = receiver.kind
        && map_path.ident.name == sym::map
        && let ExprKind::MethodCall(check_path, collection_expr, [], _) = map_receiver.kind
        && check_path.ident.name == peek_method
        && kind.is_diag_item(cx, collection_expr)
        && let ExprKind::Closure(closure) = closure_arg.kind
        && let body = cx.tcx.hir_body(closure.body)
        && cx.typeck_results().expr_ty(body.value).is_bool()
        && let Some((pop_collection, pop_span, suggestable)) = check_pop_unwrap(cx, then_block, pop_method)
        && eq_expr_value(cx, if_expr_span.ctxt(), collection_expr, pop_collection)
        && let Some(param) = body.params.first()
        && let Some(ident) = param.pat.simple_ident()
    {
        return Some(ManualPopIfPattern {
            kind,
            collection_expr,
            predicate: body.value,
            param_name: ident.name,
            if_span: if_expr_span,
            spans: MultiSpan::from(vec![if_expr_span.with_hi(cond.span.hi()), pop_span]),
            suggestable,
        });
    }

    None
}

/// Checks for `collection.<pop_method>().unwrap()` or `collection.<pop_method>().expect(..)`
/// and returns the collection expression and the span of the pop+unwrap call.
/// If the pop+unwrap is the only statement in the block, the result is marked as
/// suggestable (we can provide an automatic fix).
fn check_pop_unwrap<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    pop_method: Symbol,
) -> Option<(&'tcx Expr<'tcx>, Span, bool)> {
    let ExprKind::Block(block, _) = expr.kind else {
        return None;
    };

    let as_pop_unwrap = |expr: &Expr<'tcx>| -> Option<(&'tcx Expr<'tcx>, Span)> {
        if let ExprKind::MethodCall(unwrap_path, receiver, _, _) = expr.kind
            && matches!(
                unwrap_path.ident.name,
                sym::unwrap | sym::unwrap_unchecked | sym::expect
            )
            && let ExprKind::MethodCall(pop_path, collection_expr, [], _) = receiver.kind
            && pop_path.ident.name == pop_method
        {
            Some((collection_expr, expr.span))
        } else {
            None
        }
    };

    // Peel through an `unsafe` block for `unwrap_unchecked`.
    let peel_unsafe = |expr: &'tcx Expr<'tcx>| -> &'tcx Expr<'tcx> {
        if let ExprKind::Block(block, _) = expr.kind
            && block.rules == BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided)
            && block.stmts.is_empty()
            && let Some(inner) = block.expr
        {
            inner
        } else {
            expr
        }
    };

    // Check for single statement with the pop unwrap (not in a macro or other expression)
    // and that there are no comments or other text before or after the pop call.
    if let [stmt] = block.stmts
        && block.expr.is_none()
        && let StmtKind::Semi(stmt_expr) | StmtKind::Expr(stmt_expr) = &stmt.kind
        && !stmt_expr.span.from_expansion()
        && let Some((collection_expr, span)) = as_pop_unwrap(peel_unsafe(stmt_expr))
    {
        let span_before = block
            .span
            .with_lo(block.span.lo() + BytePos(1))
            .with_hi(stmt_expr.span.lo());
        let span_after = stmt.span.shrink_to_hi().with_hi(block.span.hi() - BytePos(1));
        let suggestable = !span_contains_non_whitespace(cx, span_before, false)
            && !span_contains_non_whitespace(cx, span_after, false);
        return Some((collection_expr, span, suggestable));
    }

    // Check if the pop unwrap is present at all
    for_each_expr_without_closures(block, |expr| {
        if let Some((collection_expr, span)) = as_pop_unwrap(expr) {
            ControlFlow::Break((collection_expr, span, false))
        } else {
            ControlFlow::Continue(())
        }
    })
}

impl<'tcx> LateLintPass<'tcx> for ManualPopIf {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let ExprKind::If(cond, then_block, None) = expr.kind else {
            return;
        };

        let in_else_clause = is_else_clause(cx.tcx, expr);

        for kind in [
            ManualPopIfKind::Vec,
            ManualPopIfKind::VecDequeBack,
            ManualPopIfKind::VecDequeFront,
            ManualPopIfKind::BinaryHeap,
        ] {
            if let Some(mut pattern) = check_is_some_and_pattern(cx, cond, then_block, expr.span, kind)
                .or_else(|| check_if_let_pattern(cx, cond, then_block, expr.span, kind))
                .or_else(|| check_let_chain_pattern(cx, cond, then_block, expr.span, kind))
                .or_else(|| check_map_unwrap_or_pattern(cx, cond, then_block, expr.span, kind))
                && self.msrv_compatible(cx, kind)
            {
                if in_else_clause {
                    pattern.suggestable = false;
                }

                pattern.emit_lint(cx);
                return;
            }
        }
    }
}
