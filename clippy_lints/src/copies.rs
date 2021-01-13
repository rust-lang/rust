use crate::utils::{both, count_eq, eq_expr_value, in_macro, search_same, SpanlessEq, SpanlessHash};
use crate::utils::{
    first_line_of_span, get_parent_expr, higher, if_sequence, indent_of, parent_node_is_if_expr, reindent_multiline,
    snippet, span_lint_and_note, span_lint_and_sugg, span_lint_and_then,
};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::{Block, Expr, HirId};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;
use std::borrow::Cow;

declare_clippy_lint! {
    /// **What it does:** Checks for consecutive `if`s with the same condition.
    ///
    /// **Why is this bad?** This is probably a copy & paste error.
    ///
    /// **Known problems:** Hopefully none.
    ///
    /// **Example:**
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
    pub IFS_SAME_COND,
    correctness,
    "consecutive `if`s with the same condition"
}

declare_clippy_lint! {
    /// **What it does:** Checks for consecutive `if`s with the same function call.
    ///
    /// **Why is this bad?** This is probably a copy & paste error.
    /// Despite the fact that function can have side effects and `if` works as
    /// intended, such an approach is implicit and can be considered a "code smell".
    ///
    /// **Known problems:** Hopefully none.
    ///
    /// **Example:**
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
    pub SAME_FUNCTIONS_IN_IF_CONDITION,
    pedantic,
    "consecutive `if`s with the same function call"
}

declare_clippy_lint! {
    /// **What it does:** Checks for `if/else` with the same body as the *then* part
    /// and the *else* part.
    ///
    /// **Why is this bad?** This is probably a copy & paste error.
    ///
    /// **Known problems:** Hopefully none.
    ///
    /// **Example:**
    /// ```ignore
    /// let foo = if … {
    ///     42
    /// } else {
    ///     42
    /// };
    /// ```
    pub IF_SAME_THEN_ELSE,
    correctness,
    "`if` with the same `then` and `else` blocks"
}

declare_clippy_lint! {
    /// **What it does:** Checks if the `if` and `else` block contain shared code that can be
    /// moved out of the blocks.
    ///
    /// **Why is this bad?** Duplicate code is less maintainable.
    ///
    /// **Known problems:** Hopefully none.
    ///
    /// **Example:**
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
    /// Could be written as:
    /// ```ignore
    /// println!("Hello World");
    /// let foo = if … {
    ///     13
    /// } else {
    ///     42
    /// };
    /// ```
    pub SHARED_CODE_IN_IF_BLOCKS,
    nursery,
    "`if` statement with shared code in all blocks"
}

declare_lint_pass!(CopyAndPaste => [
    IFS_SAME_COND,
    SAME_FUNCTIONS_IN_IF_CONDITION,
    IF_SAME_THEN_ELSE,
    SHARED_CODE_IN_IF_BLOCKS
]);

impl<'tcx> LateLintPass<'tcx> for CopyAndPaste {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !expr.span.from_expansion() {
            // skip ifs directly in else, it will be checked in the parent if
            if let Some(&Expr {
                kind: ExprKind::If(_, _, Some(ref else_expr)),
                ..
            }) = get_parent_expr(cx, expr)
            {
                if else_expr.hir_id == expr.hir_id {
                    return;
                }
            }

            let (conds, blocks) = if_sequence(expr);
            // Conditions
            lint_same_cond(cx, &conds);
            lint_same_fns_in_if_cond(cx, &conds);
            // Block duplication
            lint_same_then_else(cx, &blocks, conds.len() != blocks.len(), expr);
        }
    }
}

/// Implementation of `SHARED_CODE_IN_IF_BLOCKS` and `IF_SAME_THEN_ELSE` if the blocks are equal.
fn lint_same_then_else<'tcx>(
    cx: &LateContext<'tcx>,
    blocks: &[&Block<'tcx>],
    has_unconditional_else: bool,
    expr: &'tcx Expr<'_>,
) {
    // We only lint ifs with multiple blocks
    // TODO xFrednet 2021-01-01: Check if it's an else if block
    if blocks.len() < 2 || parent_node_is_if_expr(expr, cx) {
        return;
    }

    let has_expr = blocks[0].expr.is_some();

    // Check if each block has shared code
    let mut start_eq = usize::MAX;
    let mut end_eq = usize::MAX;
    let mut expr_eq = true;
    for win in blocks.windows(2) {
        let l_stmts = win[0].stmts;
        let r_stmts = win[1].stmts;

        let mut evaluator = SpanlessEq::new(cx);
        let current_start_eq = count_eq(&mut l_stmts.iter(), &mut r_stmts.iter(), |l, r| evaluator.eq_stmt(l, r));
        let current_end_eq = count_eq(&mut l_stmts.iter().rev(), &mut r_stmts.iter().rev(), |l, r| {
            evaluator.eq_stmt(l, r)
        });
        let block_expr_eq = both(&win[0].expr, &win[1].expr, |l, r| evaluator.eq_expr(l, r));

        // IF_SAME_THEN_ELSE
        if block_expr_eq && l_stmts.len() == r_stmts.len() && l_stmts.len() == current_start_eq {
            span_lint_and_note(
                cx,
                IF_SAME_THEN_ELSE,
                win[0].span,
                "this `if` has identical blocks",
                Some(win[1].span),
                "same as this",
            );

            return;
        } else {
            println!(
                "{:?}\n - expr_eq: {:10}, l_stmts.len(): {:10}, r_stmts.len(): {:10}",
                win[0].span,
                block_expr_eq,
                l_stmts.len(),
                r_stmts.len()
            )
        }

        start_eq = start_eq.min(current_start_eq);
        end_eq = end_eq.min(current_end_eq);
        expr_eq &= block_expr_eq;
    }

    // SHARED_CODE_IN_IF_BLOCKS prerequisites
    if !has_unconditional_else || (start_eq == 0 && end_eq == 0 && (has_expr && !expr_eq)) {
        return;
    }

    if has_expr && !expr_eq {
        end_eq = 0;
    }

    // Check if the regions are overlapping. Set `end_eq` to prevent the overlap
    let min_block_size = blocks.iter().map(|x| x.stmts.len()).min().unwrap();
    if (start_eq + end_eq) > min_block_size {
        end_eq = min_block_size - start_eq;
    }

    // Only the start is the same
    if start_eq != 0 && end_eq == 0 && (!has_expr || !expr_eq) {
        emit_shared_code_in_if_blocks_lint(cx, start_eq, 0, false, blocks, expr);
    } else if end_eq != 0 && (!has_expr || !expr_eq) {
        let block = blocks[blocks.len() - 1];
        let stmts = block.stmts.split_at(start_eq).1;
        let (block_stmts, moved_stmts) = stmts.split_at(stmts.len() - end_eq);

        // Scan block
        let mut walker = SymbolFinderVisitor::new(cx);
        for stmt in block_stmts {
            intravisit::walk_stmt(&mut walker, stmt);
        }
        let mut block_defs = walker.defs;

        // Scan moved stmts
        let mut moved_start: Option<usize> = None;
        let mut walker = SymbolFinderVisitor::new(cx);
        for (index, stmt) in moved_stmts.iter().enumerate() {
            intravisit::walk_stmt(&mut walker, stmt);

            for value in &walker.uses {
                // Well we can't move this and all prev statements. So reset
                if block_defs.contains(&value) {
                    moved_start = Some(index + 1);
                    walker.defs.drain().for_each(|x| {
                        block_defs.insert(x);
                    });
                }
            }

            walker.uses.clear();
        }

        if let Some(moved_start) = moved_start {
            end_eq -= moved_start;
        }

        let end_linable = if let Some(expr) = block.expr {
            intravisit::walk_expr(&mut walker, expr);
            walker.uses.iter().any(|x| !block_defs.contains(x))
        } else if end_eq == 0 {
            false
        } else {
            true
        };

        emit_shared_code_in_if_blocks_lint(cx, start_eq, end_eq, end_linable, blocks, expr);
    }
}

fn emit_shared_code_in_if_blocks_lint(
    cx: &LateContext<'tcx>,
    start_stmts: usize,
    end_stmts: usize,
    lint_end: bool,
    blocks: &[&Block<'tcx>],
    if_expr: &'tcx Expr<'_>,
) {
    if start_stmts == 0 && !lint_end {
        return;
    }

    // (help, span, suggestion)
    let mut suggestions: Vec<(&str, Span, String)> = vec![];

    if start_stmts > 0 {
        let block = blocks[0];
        let span_start = first_line_of_span(cx, if_expr.span).shrink_to_lo();
        let span_end = block.stmts[start_stmts - 1].span.source_callsite();

        let cond_span = first_line_of_span(cx, if_expr.span).until(block.span);
        let cond_snippet = reindent_multiline(snippet(cx, cond_span, "_"), false, None);
        let cond_indent = indent_of(cx, cond_span);
        let moved_span = block.stmts[0].span.source_callsite().to(span_end);
        let moved_snippet = reindent_multiline(snippet(cx, moved_span, "_"), true, None);
        let suggestion = moved_snippet.to_string() + "\n" + &cond_snippet + "{";
        let suggestion = reindent_multiline(Cow::Borrowed(&suggestion), true, cond_indent);

        let span = span_start.to(span_end);
        suggestions.push(("START HELP", span, suggestion.to_string()));
    }

    if lint_end {
        let block = blocks[blocks.len() - 1];
        let span_end = block.span.shrink_to_hi();

        let moved_start = if end_stmts == 0 && block.expr.is_some() {
            block.expr.unwrap().span
        } else {
            block.stmts[block.stmts.len() - end_stmts].span
        }
        .source_callsite();
        let moved_end = if let Some(expr) = block.expr {
            expr.span
        } else {
            block.stmts[block.stmts.len() - 1].span
        }
        .source_callsite();

        let moved_span = moved_start.to(moved_end);
        let moved_snipped = reindent_multiline(snippet(cx, moved_span, "_"), true, None);
        let indent = indent_of(cx, if_expr.span.shrink_to_hi());
        let suggestion = "}\n".to_string() + &moved_snipped;
        let suggestion = reindent_multiline(Cow::Borrowed(&suggestion), true, indent);

        let span = moved_start.to(span_end);
        suggestions.push(("END_RANGE", span, suggestion.to_string()));
    }

    if suggestions.len() == 1 {
        let (_, span, sugg) = &suggestions[0];
        span_lint_and_sugg(
            cx,
            SHARED_CODE_IN_IF_BLOCKS,
            *span,
            "All code blocks contain the same code",
            "Consider moving the statements out like this",
            sugg.clone(),
            Applicability::Unspecified,
        );
    } else {
        span_lint_and_then(
            cx,
            SHARED_CODE_IN_IF_BLOCKS,
            if_expr.span,
            "All if blocks contain the same code",
            move |diag| {
                for (help, span, sugg) in suggestions {
                    diag.span_suggestion(span, help, sugg, Applicability::Unspecified);
                }
            },
        );
    }
}

pub struct SymbolFinderVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    defs: FxHashSet<HirId>,
    uses: FxHashSet<HirId>,
}

impl<'a, 'tcx> SymbolFinderVisitor<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>) -> Self {
        SymbolFinderVisitor {
            cx,
            defs: FxHashSet::default(),
            uses: FxHashSet::default(),
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for SymbolFinderVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::All(self.cx.tcx.hir())
    }

    fn visit_local(&mut self, l: &'tcx rustc_hir::Local<'tcx>) {
        let local_id = l.pat.hir_id;
        self.defs.insert(local_id);
        if let Some(expr) = l.init {
            intravisit::walk_expr(self, expr);
        }
    }

    fn visit_qpath(&mut self, qpath: &'tcx rustc_hir::QPath<'tcx>, id: HirId, _span: rustc_span::Span) {
        if let rustc_hir::QPath::Resolved(_, ref path) = *qpath {
            if path.segments.len() == 1 {
                if let rustc_hir::def::Res::Local(var) = self.cx.qpath_res(qpath, id) {
                    self.uses.insert(var);
                }
            }
        }
    }
}

/// Implementation of `IFS_SAME_COND`.
fn lint_same_cond(cx: &LateContext<'_>, conds: &[&Expr<'_>]) {
    let hash: &dyn Fn(&&Expr<'_>) -> u64 = &|expr| -> u64 {
        let mut h = SpanlessHash::new(cx);
        h.hash_expr(expr);
        h.finish()
    };

    let eq: &dyn Fn(&&Expr<'_>, &&Expr<'_>) -> bool = &|&lhs, &rhs| -> bool { eq_expr_value(cx, lhs, rhs) };

    for (i, j) in search_same(conds, hash, eq) {
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
    let hash: &dyn Fn(&&Expr<'_>) -> u64 = &|expr| -> u64 {
        let mut h = SpanlessHash::new(cx);
        h.hash_expr(expr);
        h.finish()
    };

    let eq: &dyn Fn(&&Expr<'_>, &&Expr<'_>) -> bool = &|&lhs, &rhs| -> bool {
        // Do not lint if any expr originates from a macro
        if in_macro(lhs.span) || in_macro(rhs.span) {
            return false;
        }
        // Do not spawn warning if `IFS_SAME_COND` already produced it.
        if eq_expr_value(cx, lhs, rhs) {
            return false;
        }
        SpanlessEq::new(cx).eq_expr(lhs, rhs)
    };

    for (i, j) in search_same(conds, hash, eq) {
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
