use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::res::MaybeDef;
use clippy_utils::source::snippet_with_context;
use clippy_utils::visitors::is_local_used;
use clippy_utils::{eq_expr_value, is_else_clause, is_lang_item_or_ctor, peel_blocks_with_stmt, sym};
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, LangItem, PatKind, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TyCtxt;
use rustc_session::impl_lint_pass;
use rustc_span::{Span, Symbol};
use std::fmt;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for code to be replaced by `pop_if` methods.
    ///
    /// ### Why is this bad?
    /// Using `pop_if` is more concise and idiomatic.
    ///
    /// ### Known issues
    /// Currently, the lint does not handle the case where the
    /// `if` condition is part of an `else if` branch.
    ///
    /// The lint also does not handle the case where
    /// the popped value is assigned and used.
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
    fn check_method(self) -> Symbol {
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
}

impl ManualPopIfPattern<'_> {
    fn emit_lint(&self, cx: &LateContext<'_>) {
        let mut app = Applicability::MachineApplicable;
        let ctxt = self.if_span.ctxt();
        let collection_snippet = snippet_with_context(cx, self.collection_expr.span, ctxt, "..", &mut app).0;
        let predicate_snippet = snippet_with_context(cx, self.predicate.span, ctxt, "..", &mut app).0;
        let param_name = self.param_name;
        let pop_if_method = self.kind.pop_if_method();

        let suggestion = format!("{collection_snippet}.{pop_if_method}(|{param_name}| {predicate_snippet});");

        span_lint_and_sugg(
            cx,
            MANUAL_POP_IF,
            self.if_span,
            format!("manual implementation of {}", self.kind),
            "try",
            suggestion,
            app,
        );
    }
}

fn pop_value_is_used(then_block: &Expr<'_>) -> bool {
    let ExprKind::Block(block, _) = then_block.kind else {
        return true;
    };

    if block.expr.is_some() {
        return true;
    }

    match block.stmts {
        [stmt] => !matches!(stmt.kind, StmtKind::Semi(_) | StmtKind::Item(_)),
        [.., last] => matches!(last.kind, StmtKind::Expr(_)),
        [] => false,
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
    if pop_value_is_used(then_block) {
        return None;
    }

    let check_method = kind.check_method();
    let pop_method = kind.pop_method();

    if let ExprKind::MethodCall(path, receiver, [closure_arg], _) = cond.kind
        && path.ident.name == sym::is_some_and
        && let ExprKind::MethodCall(check_path, collection_expr, [], _) = receiver.kind
        && check_path.ident.name == check_method
        && kind.is_diag_item(cx, collection_expr)
        && let ExprKind::Closure(closure) = closure_arg.kind
        && let body = cx.tcx.hir_body(closure.body)
        && let Some((pop_collection, _pop_span)) = check_pop_unwrap(then_block, pop_method)
        && eq_expr_value(cx, collection_expr, pop_collection)
        && let Some(param) = body.params.first()
        && let Some(ident) = param.pat.simple_ident()
    {
        return Some(ManualPopIfPattern {
            kind,
            collection_expr,
            predicate: body.value,
            param_name: ident.name,
            if_span: if_expr_span,
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
    let check_method = kind.check_method();
    let pop_method = kind.pop_method();

    if let ExprKind::Let(let_expr) = cond.kind
        && let PatKind::TupleStruct(qpath, [binding_pat], _) = let_expr.pat.kind
    {
        let res = cx.qpath_res(&qpath, let_expr.pat.hir_id);

        if let Some(def_id) = res.opt_def_id()
            && is_lang_item_or_ctor(cx, def_id, LangItem::OptionSome)
            && let PatKind::Binding(_, binding_id, binding_name, _) = binding_pat.kind
            && let ExprKind::MethodCall(path, collection_expr, [], _) = let_expr.init.kind
            && path.ident.name == check_method
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
                && !pop_value_is_used(inner_then)
                && let Some((pop_collection, _pop_span)) = check_pop_unwrap(inner_then, pop_method)
                && eq_expr_value(cx, collection_expr, pop_collection)
            {
                return Some(ManualPopIfPattern {
                    kind,
                    collection_expr,
                    predicate: inner_cond,
                    param_name: binding_name.name,
                    if_span: if_expr_span,
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
    if pop_value_is_used(then_block) {
        return None;
    }

    let check_method = kind.check_method();
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
            && path.ident.name == check_method
            && kind.is_diag_item(cx, collection_expr)
            && is_local_used(cx, right, binding_id)
            && !pop_value_is_used(then_block)
            && let Some((pop_collection, _pop_span)) = check_pop_unwrap(then_block, pop_method)
            && eq_expr_value(cx, collection_expr, pop_collection)
        {
            return Some(ManualPopIfPattern {
                kind,
                collection_expr,
                predicate: right,
                param_name: binding_name.name,
                if_span: if_expr_span,
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
    if pop_value_is_used(then_block) {
        return None;
    }

    let check_method = kind.check_method();
    let pop_method = kind.pop_method();

    if let ExprKind::MethodCall(unwrap_path, receiver, [default_arg], _) = cond.kind
        && unwrap_path.ident.name == sym::unwrap_or
        && matches!(default_arg.kind, ExprKind::Lit(lit) if matches!(lit.node, LitKind::Bool(false)))
        && let ExprKind::MethodCall(map_path, map_receiver, [closure_arg], _) = receiver.kind
        && map_path.ident.name == sym::map
        && let ExprKind::MethodCall(check_path, collection_expr, [], _) = map_receiver.kind
        && check_path.ident.name == check_method
        && kind.is_diag_item(cx, collection_expr)
        && let ExprKind::Closure(closure) = closure_arg.kind
        && let body = cx.tcx.hir_body(closure.body)
        && cx.typeck_results().expr_ty(body.value).is_bool()
        && let Some((pop_collection, _pop_span)) = check_pop_unwrap(then_block, pop_method)
        && eq_expr_value(cx, collection_expr, pop_collection)
        && let Some(param) = body.params.first()
        && let Some(ident) = param.pat.simple_ident()
    {
        return Some(ManualPopIfPattern {
            kind,
            collection_expr,
            predicate: body.value,
            param_name: ident.name,
            if_span: if_expr_span,
        });
    }

    None
}

/// Checks for `collection.<pop_method>().unwrap()` or `collection.<pop_method>().expect(..)`
/// and returns the collection and the span of the peeled expr
fn check_pop_unwrap<'tcx>(expr: &'tcx Expr<'_>, pop_method: Symbol) -> Option<(&'tcx Expr<'tcx>, Span)> {
    let inner_expr = peel_blocks_with_stmt(expr);

    if let ExprKind::MethodCall(unwrap_path, receiver, _, _) = inner_expr.kind
        && matches!(unwrap_path.ident.name, sym::unwrap | sym::expect)
        && let ExprKind::MethodCall(pop_path, collection_expr, [], _) = receiver.kind
        && pop_path.ident.name == pop_method
    {
        return Some((collection_expr, inner_expr.span));
    }

    None
}

impl<'tcx> LateLintPass<'tcx> for ManualPopIf {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let ExprKind::If(cond, then_block, None) = expr.kind else {
            return;
        };

        // Do not lint if we are in an else-if branch.
        if is_else_clause(cx.tcx, expr) {
            return;
        }

        for kind in [
            ManualPopIfKind::Vec,
            ManualPopIfKind::VecDequeBack,
            ManualPopIfKind::VecDequeFront,
            ManualPopIfKind::BinaryHeap,
        ] {
            if let Some(pattern) = check_is_some_and_pattern(cx, cond, then_block, expr.span, kind)
                .or_else(|| check_if_let_pattern(cx, cond, then_block, expr.span, kind))
                .or_else(|| check_let_chain_pattern(cx, cond, then_block, expr.span, kind))
                .or_else(|| check_map_unwrap_or_pattern(cx, cond, then_block, expr.span, kind))
                && self.msrv_compatible(cx, kind)
            {
                pattern.emit_lint(cx);
                return;
            }
        }
    }
}
