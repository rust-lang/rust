use clippy_config::Conf;
use clippy_utils::diagnostics::{span_lint_and_note, span_lint_and_then};
use clippy_utils::source::{IntoSpan, SpanRangeExt, first_line_of_span, indent_of, reindent_multiline, snippet};
use clippy_utils::ty::{InteriorMut, needs_ordered_drop};
use clippy_utils::visitors::for_each_expr_without_closures;
use clippy_utils::{
    ContainsName, HirEqInterExpr, SpanlessEq, capture_local_usage, eq_expr_value, find_binding_init,
    get_enclosing_block, hash_expr, hash_stmt, if_sequence, is_else_clause, is_lint_allowed, path_to_local,
    search_same,
};
use core::iter;
use core::ops::ControlFlow;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Block, Expr, ExprKind, HirId, HirIdSet, Stmt, StmtKind, intravisit};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TyCtxt;
use rustc_session::impl_lint_pass;
use rustc_span::hygiene::walk_chain;
use rustc_span::source_map::SourceMap;
use rustc_span::{Span, Symbol};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for consecutive `if`s with the same condition.
    ///
    /// ### Why is this bad?
    /// This is probably a copy & paste error.
    ///
    /// ### Example
    /// ```ignore
    /// if a == b {
    ///     …
    /// } else if a == b {
    ///     …
    /// }
    /// ```
    ///
    /// Note that this lint ignores all conditions with a function call as it could
    /// have side effects:
    ///
    /// ```ignore
    /// if foo() {
    ///     …
    /// } else if foo() { // not linted
    ///     …
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub IFS_SAME_COND,
    correctness,
    "consecutive `if`s with the same condition"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for consecutive `if`s with the same function call.
    ///
    /// ### Why is this bad?
    /// This is probably a copy & paste error.
    /// Despite the fact that function can have side effects and `if` works as
    /// intended, such an approach is implicit and can be considered a "code smell".
    ///
    /// ### Example
    /// ```ignore
    /// if foo() == bar {
    ///     …
    /// } else if foo() == bar {
    ///     …
    /// }
    /// ```
    ///
    /// This probably should be:
    /// ```ignore
    /// if foo() == bar {
    ///     …
    /// } else if foo() == baz {
    ///     …
    /// }
    /// ```
    ///
    /// or if the original code was not a typo and called function mutates a state,
    /// consider move the mutation out of the `if` condition to avoid similarity to
    /// a copy & paste error:
    ///
    /// ```ignore
    /// let first = foo();
    /// if first == bar {
    ///     …
    /// } else {
    ///     let second = foo();
    ///     if second == bar {
    ///     …
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.41.0"]
    pub SAME_FUNCTIONS_IN_IF_CONDITION,
    pedantic,
    "consecutive `if`s with the same function call"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `if/else` with the same body as the *then* part
    /// and the *else* part.
    ///
    /// ### Why is this bad?
    /// This is probably a copy & paste error.
    ///
    /// ### Example
    /// ```ignore
    /// let foo = if … {
    ///     42
    /// } else {
    ///     42
    /// };
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub IF_SAME_THEN_ELSE,
    style,
    "`if` with the same `then` and `else` blocks"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks if the `if` and `else` block contain shared code that can be
    /// moved out of the blocks.
    ///
    /// ### Why is this bad?
    /// Duplicate code is less maintainable.
    ///
    /// ### Example
    /// ```ignore
    /// let foo = if … {
    ///     println!("Hello World");
    ///     13
    /// } else {
    ///     println!("Hello World");
    ///     42
    /// };
    /// ```
    ///
    /// Use instead:
    /// ```ignore
    /// println!("Hello World");
    /// let foo = if … {
    ///     13
    /// } else {
    ///     42
    /// };
    /// ```
    #[clippy::version = "1.53.0"]
    pub BRANCHES_SHARING_CODE,
    nursery,
    "`if` statement with shared code in all blocks"
}

pub struct CopyAndPaste<'tcx> {
    interior_mut: InteriorMut<'tcx>,
}

impl<'tcx> CopyAndPaste<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, conf: &'static Conf) -> Self {
        Self {
            interior_mut: InteriorMut::new(tcx, &conf.ignore_interior_mutability),
        }
    }
}

impl_lint_pass!(CopyAndPaste<'_> => [
    IFS_SAME_COND,
    SAME_FUNCTIONS_IN_IF_CONDITION,
    IF_SAME_THEN_ELSE,
    BRANCHES_SHARING_CODE
]);

impl<'tcx> LateLintPass<'tcx> for CopyAndPaste<'tcx> {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !expr.span.from_expansion() && matches!(expr.kind, ExprKind::If(..)) && !is_else_clause(cx.tcx, expr) {
            let (conds, blocks) = if_sequence(expr);
            lint_same_cond(cx, &conds, &mut self.interior_mut);
            lint_same_fns_in_if_cond(cx, &conds);
            let all_same =
                !is_lint_allowed(cx, IF_SAME_THEN_ELSE, expr.hir_id) && lint_if_same_then_else(cx, &conds, &blocks);
            if !all_same && conds.len() != blocks.len() {
                lint_branches_sharing_code(cx, &conds, &blocks, expr);
            }
        }
    }
}

/// Checks if the given expression is a let chain.
fn contains_let(e: &Expr<'_>) -> bool {
    match e.kind {
        ExprKind::Let(..) => true,
        ExprKind::Binary(op, lhs, rhs) if op.node == BinOpKind::And => {
            matches!(lhs.kind, ExprKind::Let(..)) || contains_let(rhs)
        },
        _ => false,
    }
}

fn lint_if_same_then_else(cx: &LateContext<'_>, conds: &[&Expr<'_>], blocks: &[&Block<'_>]) -> bool {
    let mut eq = SpanlessEq::new(cx);
    blocks
        .array_windows::<2>()
        .enumerate()
        .fold(true, |all_eq, (i, &[lhs, rhs])| {
            if eq.eq_block(lhs, rhs) && !contains_let(conds[i]) && conds.get(i + 1).is_none_or(|e| !contains_let(e)) {
                span_lint_and_note(
                    cx,
                    IF_SAME_THEN_ELSE,
                    lhs.span,
                    "this `if` has identical blocks",
                    Some(rhs.span),
                    "same as this",
                );
                all_eq
            } else {
                false
            }
        })
}

fn lint_branches_sharing_code<'tcx>(
    cx: &LateContext<'tcx>,
    conds: &[&'tcx Expr<'_>],
    blocks: &[&'tcx Block<'_>],
    expr: &'tcx Expr<'_>,
) {
    // We only lint ifs with multiple blocks
    let &[first_block, ref blocks @ ..] = blocks else {
        return;
    };
    let &[.., last_block] = blocks else {
        return;
    };

    let res = scan_block_for_eq(cx, conds, first_block, blocks);
    let sm = cx.tcx.sess.source_map();
    let start_suggestion = res.start_span(first_block, sm).map(|span| {
        let first_line_span = first_line_of_span(cx, expr.span);
        let replace_span = first_line_span.with_hi(span.hi());
        let cond_span = first_line_span.until(first_block.span);
        let cond_snippet = reindent_multiline(&snippet(cx, cond_span, "_"), false, None);
        let cond_indent = indent_of(cx, cond_span);
        let moved_snippet = reindent_multiline(&snippet(cx, span, "_"), true, None);
        let suggestion = moved_snippet.to_string() + "\n" + &cond_snippet + "{";
        let suggestion = reindent_multiline(&suggestion, true, cond_indent);
        (replace_span, suggestion.to_string())
    });
    let end_suggestion = res.end_span(last_block, sm).map(|span| {
        let moved_snipped = reindent_multiline(&snippet(cx, span, "_"), true, None);
        let indent = indent_of(cx, expr.span.shrink_to_hi());
        let suggestion = "}\n".to_string() + &moved_snipped;
        let suggestion = reindent_multiline(&suggestion, true, indent);

        let span = span.with_hi(last_block.span.hi());
        // Improve formatting if the inner block has indentation (i.e. normal Rust formatting)
        let span = span
            .map_range(cx, |src, range| {
                (range.start > 4 && src.get(range.start - 4..range.start)? == "    ")
                    .then_some(range.start - 4..range.end)
            })
            .map_or(span, |range| range.with_ctxt(span.ctxt()));
        (span, suggestion.to_string())
    });

    let (span, msg, end_span) = match (&start_suggestion, &end_suggestion) {
        (&Some((span, _)), &Some((end_span, _))) => (
            span,
            "all if blocks contain the same code at both the start and the end",
            Some(end_span),
        ),
        (&Some((span, _)), None) => (span, "all if blocks contain the same code at the start", None),
        (None, &Some((span, _))) => (span, "all if blocks contain the same code at the end", None),
        (None, None) => return,
    };
    span_lint_and_then(cx, BRANCHES_SHARING_CODE, span, msg, |diag| {
        if let Some(span) = end_span {
            diag.span_note(span, "this code is shared at the end");
        }
        if let Some((span, sugg)) = start_suggestion {
            diag.span_suggestion(
                span,
                "consider moving these statements before the if",
                sugg,
                Applicability::Unspecified,
            );
        }
        if let Some((span, sugg)) = end_suggestion {
            diag.span_suggestion(
                span,
                "consider moving these statements after the if",
                sugg,
                Applicability::Unspecified,
            );
            if !cx.typeck_results().expr_ty(expr).is_unit() {
                diag.note("the end suggestion probably needs some adjustments to use the expression result correctly");
            }
        }
        if check_for_warn_of_moved_symbol(cx, &res.moved_locals, expr) {
            diag.warn("some moved values might need to be renamed to avoid wrong references");
        }
    });
}

struct BlockEq {
    /// The end of the range of equal stmts at the start.
    start_end_eq: usize,
    /// The start of the range of equal stmts at the end.
    end_begin_eq: Option<usize>,
    /// The name and id of every local which can be moved at the beginning and the end.
    moved_locals: Vec<(HirId, Symbol)>,
}
impl BlockEq {
    fn start_span(&self, b: &Block<'_>, sm: &SourceMap) -> Option<Span> {
        match &b.stmts[..self.start_end_eq] {
            [first, .., last] => Some(sm.stmt_span(first.span, b.span).to(sm.stmt_span(last.span, b.span))),
            [s] => Some(sm.stmt_span(s.span, b.span)),
            [] => None,
        }
    }

    fn end_span(&self, b: &Block<'_>, sm: &SourceMap) -> Option<Span> {
        match (&b.stmts[b.stmts.len() - self.end_begin_eq?..], b.expr) {
            ([first, .., last], None) => Some(sm.stmt_span(first.span, b.span).to(sm.stmt_span(last.span, b.span))),
            ([first, ..], Some(last)) => Some(sm.stmt_span(first.span, b.span).to(sm.stmt_span(last.span, b.span))),
            ([s], None) => Some(sm.stmt_span(s.span, b.span)),
            ([], Some(e)) => Some(walk_chain(e.span, b.span.ctxt())),
            ([], None) => None,
        }
    }
}

/// If the statement is a local, checks if the bound names match the expected list of names.
fn eq_binding_names(s: &Stmt<'_>, names: &[(HirId, Symbol)]) -> bool {
    if let StmtKind::Let(l) = s.kind {
        let mut i = 0usize;
        let mut res = true;
        l.pat.each_binding_or_first(&mut |_, _, _, name| {
            if names.get(i).is_some_and(|&(_, n)| n == name.name) {
                i += 1;
            } else {
                res = false;
            }
        });
        res && i == names.len()
    } else {
        false
    }
}

/// Checks if the statement modifies or moves any of the given locals.
fn modifies_any_local<'tcx>(cx: &LateContext<'tcx>, s: &'tcx Stmt<'_>, locals: &HirIdSet) -> bool {
    for_each_expr_without_closures(s, |e| {
        if let Some(id) = path_to_local(e)
            && locals.contains(&id)
            && !capture_local_usage(cx, e).is_imm_ref()
        {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .is_some()
}

/// Checks if the given statement should be considered equal to the statement in the same position
/// for each block.
fn eq_stmts(
    stmt: &Stmt<'_>,
    blocks: &[&Block<'_>],
    get_stmt: impl for<'a> Fn(&'a Block<'a>) -> Option<&'a Stmt<'a>>,
    eq: &mut HirEqInterExpr<'_, '_, '_>,
    moved_bindings: &mut Vec<(HirId, Symbol)>,
) -> bool {
    (if let StmtKind::Let(l) = stmt.kind {
        let old_count = moved_bindings.len();
        l.pat.each_binding_or_first(&mut |_, id, _, name| {
            moved_bindings.push((id, name.name));
        });
        let new_bindings = &moved_bindings[old_count..];
        blocks
            .iter()
            .all(|b| get_stmt(b).is_some_and(|s| eq_binding_names(s, new_bindings)))
    } else {
        true
    }) && blocks.iter().all(|b| get_stmt(b).is_some_and(|s| eq.eq_stmt(s, stmt)))
}

#[expect(clippy::too_many_lines)]
fn scan_block_for_eq<'tcx>(
    cx: &LateContext<'tcx>,
    conds: &[&'tcx Expr<'_>],
    block: &'tcx Block<'_>,
    blocks: &[&'tcx Block<'_>],
) -> BlockEq {
    let mut eq = SpanlessEq::new(cx);
    let mut eq = eq.inter_expr();
    let mut moved_locals = Vec::new();

    let mut cond_locals = HirIdSet::default();
    for &cond in conds {
        let _: Option<!> = for_each_expr_without_closures(cond, |e| {
            if let Some(id) = path_to_local(e) {
                cond_locals.insert(id);
            }
            ControlFlow::Continue(())
        });
    }

    let mut local_needs_ordered_drop = false;
    let start_end_eq = block
        .stmts
        .iter()
        .enumerate()
        .find(|&(i, stmt)| {
            if let StmtKind::Let(l) = stmt.kind
                && needs_ordered_drop(cx, cx.typeck_results().node_type(l.hir_id))
            {
                local_needs_ordered_drop = true;
                return true;
            }
            modifies_any_local(cx, stmt, &cond_locals)
                || !eq_stmts(stmt, blocks, |b| b.stmts.get(i), &mut eq, &mut moved_locals)
        })
        .map_or(block.stmts.len(), |(i, _)| i);

    if local_needs_ordered_drop {
        return BlockEq {
            start_end_eq,
            end_begin_eq: None,
            moved_locals,
        };
    }

    // Walk backwards through the final expression/statements so long as their hashes are equal. Note
    // `SpanlessHash` treats all local references as equal allowing locals declared earlier in the block
    // to match those in other blocks. e.g. If each block ends with the following the hash value will be
    // the same even though each `x` binding will have a different `HirId`:
    //     let x = foo();
    //     x + 50
    let expr_hash_eq = if let Some(e) = block.expr {
        let hash = hash_expr(cx, e);
        blocks.iter().all(|b| b.expr.is_some_and(|e| hash_expr(cx, e) == hash))
    } else {
        blocks.iter().all(|b| b.expr.is_none())
    };
    if !expr_hash_eq {
        return BlockEq {
            start_end_eq,
            end_begin_eq: None,
            moved_locals,
        };
    }
    let end_search_start = block.stmts[start_end_eq..]
        .iter()
        .rev()
        .enumerate()
        .find(|&(offset, stmt)| {
            let hash = hash_stmt(cx, stmt);
            blocks.iter().any(|b| {
                b.stmts
                    // the bounds check will catch the underflow
                    .get(b.stmts.len().wrapping_sub(offset + 1))
                    .is_none_or(|s| hash != hash_stmt(cx, s))
            })
        })
        .map_or(block.stmts.len() - start_end_eq, |(i, _)| i);

    let moved_locals_at_start = moved_locals.len();
    let mut i = end_search_start;
    let end_begin_eq = block.stmts[block.stmts.len() - end_search_start..]
        .iter()
        .zip(iter::repeat_with(move || {
            let x = i;
            i -= 1;
            x
        }))
        .fold(end_search_start, |init, (stmt, offset)| {
            if eq_stmts(
                stmt,
                blocks,
                |b| b.stmts.get(b.stmts.len() - offset),
                &mut eq,
                &mut moved_locals,
            ) {
                init
            } else {
                // Clear out all locals seen at the end so far. None of them can be moved.
                let stmts = &blocks[0].stmts;
                for stmt in &stmts[stmts.len() - init..=stmts.len() - offset] {
                    if let StmtKind::Let(l) = stmt.kind {
                        l.pat.each_binding_or_first(&mut |_, id, _, _| {
                            // FIXME(rust/#120456) - is `swap_remove` correct?
                            eq.locals.swap_remove(&id);
                        });
                    }
                }
                moved_locals.truncate(moved_locals_at_start);
                offset - 1
            }
        });
    if let Some(e) = block.expr {
        for block in blocks {
            if block.expr.is_some_and(|expr| !eq.eq_expr(expr, e)) {
                moved_locals.truncate(moved_locals_at_start);
                return BlockEq {
                    start_end_eq,
                    end_begin_eq: None,
                    moved_locals,
                };
            }
        }
    }

    BlockEq {
        start_end_eq,
        end_begin_eq: Some(end_begin_eq),
        moved_locals,
    }
}

fn check_for_warn_of_moved_symbol(cx: &LateContext<'_>, symbols: &[(HirId, Symbol)], if_expr: &Expr<'_>) -> bool {
    get_enclosing_block(cx, if_expr.hir_id).is_some_and(|block| {
        let ignore_span = block.span.shrink_to_lo().to(if_expr.span);

        symbols
            .iter()
            .filter(|&&(_, name)| !name.as_str().starts_with('_'))
            .any(|&(_, name)| {
                let mut walker = ContainsName { name, cx };

                // Scan block
                let mut res = block
                    .stmts
                    .iter()
                    .filter(|stmt| !ignore_span.overlaps(stmt.span))
                    .try_for_each(|stmt| intravisit::walk_stmt(&mut walker, stmt));

                if let Some(expr) = block.expr
                    && res.is_continue()
                {
                    res = intravisit::walk_expr(&mut walker, expr);
                }

                res.is_break()
            })
    })
}

fn method_caller_is_mutable<'tcx>(
    cx: &LateContext<'tcx>,
    caller_expr: &Expr<'_>,
    interior_mut: &mut InteriorMut<'tcx>,
) -> bool {
    let caller_ty = cx.typeck_results().expr_ty(caller_expr);

    interior_mut.is_interior_mut_ty(cx, caller_ty)
        || caller_ty.is_mutable_ptr()
        // `find_binding_init` will return the binding iff its not mutable
        || path_to_local(caller_expr)
            .and_then(|hid| find_binding_init(cx, hid))
            .is_none()
}

/// Implementation of `IFS_SAME_COND`.
fn lint_same_cond<'tcx>(cx: &LateContext<'tcx>, conds: &[&Expr<'_>], interior_mut: &mut InteriorMut<'tcx>) {
    for (i, j) in search_same(
        conds,
        |e| hash_expr(cx, e),
        |lhs, rhs| {
            // Ignore eq_expr side effects iff one of the expression kind is a method call
            // and the caller is not a mutable, including inner mutable type.
            if let ExprKind::MethodCall(_, caller, _, _) = lhs.kind {
                if method_caller_is_mutable(cx, caller, interior_mut) {
                    false
                } else {
                    SpanlessEq::new(cx).eq_expr(lhs, rhs)
                }
            } else {
                eq_expr_value(cx, lhs, rhs)
            }
        },
    ) {
        span_lint_and_note(
            cx,
            IFS_SAME_COND,
            j.span,
            "this `if` has the same condition as a previous `if`",
            Some(i.span),
            "same as this",
        );
    }
}

/// Implementation of `SAME_FUNCTIONS_IN_IF_CONDITION`.
fn lint_same_fns_in_if_cond(cx: &LateContext<'_>, conds: &[&Expr<'_>]) {
    let eq: &dyn Fn(&&Expr<'_>, &&Expr<'_>) -> bool = &|&lhs, &rhs| -> bool {
        // Do not lint if any expr originates from a macro
        if lhs.span.from_expansion() || rhs.span.from_expansion() {
            return false;
        }
        // Do not spawn warning if `IFS_SAME_COND` already produced it.
        if eq_expr_value(cx, lhs, rhs) {
            return false;
        }
        SpanlessEq::new(cx).eq_expr(lhs, rhs)
    };

    for (i, j) in search_same(conds, |e| hash_expr(cx, e), eq) {
        span_lint_and_note(
            cx,
            SAME_FUNCTIONS_IN_IF_CONDITION,
            j.span,
            "this `if` has the same function call as a previous `if`",
            Some(i.span),
            "same as this",
        );
    }
}
