use clippy_utils::diagnostics::{span_lint_and_note, span_lint_and_then};
use clippy_utils::source::{first_line_of_span, indent_of, reindent_multiline, snippet, snippet_opt};
use clippy_utils::{
    both, count_eq, eq_expr_value, get_enclosing_block, get_parent_expr, if_sequence, in_macro, is_else_clause,
    is_lint_allowed, search_same, ContainsName, SpanlessEq, SpanlessHash,
};
use if_chain::if_chain;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::{Block, Expr, ExprKind, HirId};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{source_map::Span, symbol::Symbol, BytePos};
use std::borrow::Cow;

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
    pub IF_SAME_THEN_ELSE,
    correctness,
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
    /// ### Known problems
    /// * The lint doesn't check if the moved expressions modify values that are beeing used in
    ///   the if condition. The suggestion can in that case modify the behavior of the program.
    ///   See [rust-clippy#7452](https://github.com/rust-lang/rust-clippy/issues/7452)
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
    /// Could be written as:
    /// ```ignore
    /// println!("Hello World");
    /// let foo = if … {
    ///     13
    /// } else {
    ///     42
    /// };
    /// ```
    pub BRANCHES_SHARING_CODE,
    nursery,
    "`if` statement with shared code in all blocks"
}

declare_lint_pass!(CopyAndPaste => [
    IFS_SAME_COND,
    SAME_FUNCTIONS_IN_IF_CONDITION,
    IF_SAME_THEN_ELSE,
    BRANCHES_SHARING_CODE
]);

impl<'tcx> LateLintPass<'tcx> for CopyAndPaste {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !expr.span.from_expansion() {
            if let ExprKind::If(_, _, _) = expr.kind {
                // skip ifs directly in else, it will be checked in the parent if
                if let Some(&Expr {
                    kind: ExprKind::If(_, _, Some(else_expr)),
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
                lint_same_then_else(cx, &blocks, conds.len() == blocks.len(), expr);
            }
        }
    }
}

/// Implementation of `BRANCHES_SHARING_CODE` and `IF_SAME_THEN_ELSE` if the blocks are equal.
fn lint_same_then_else<'tcx>(
    cx: &LateContext<'tcx>,
    blocks: &[&Block<'tcx>],
    has_conditional_else: bool,
    expr: &'tcx Expr<'_>,
) {
    // We only lint ifs with multiple blocks
    if blocks.len() < 2 || is_else_clause(cx.tcx, expr) {
        return;
    }

    // Check if each block has shared code
    let has_expr = blocks[0].expr.is_some();

    let (start_eq, mut end_eq, expr_eq) = if let Some(block_eq) = scan_block_for_eq(cx, blocks) {
        (block_eq.start_eq, block_eq.end_eq, block_eq.expr_eq)
    } else {
        return;
    };

    // BRANCHES_SHARING_CODE prerequisites
    if has_conditional_else || (start_eq == 0 && end_eq == 0 && (has_expr && !expr_eq)) {
        return;
    }

    // Only the start is the same
    if start_eq != 0 && end_eq == 0 && (!has_expr || !expr_eq) {
        let block = blocks[0];
        let start_stmts = block.stmts.split_at(start_eq).0;

        let mut start_walker = UsedValueFinderVisitor::new(cx);
        for stmt in start_stmts {
            intravisit::walk_stmt(&mut start_walker, stmt);
        }

        emit_branches_sharing_code_lint(
            cx,
            start_eq,
            0,
            false,
            check_for_warn_of_moved_symbol(cx, &start_walker.def_symbols, expr),
            blocks,
            expr,
        );
    } else if end_eq != 0 || (has_expr && expr_eq) {
        let block = blocks[blocks.len() - 1];
        let (start_stmts, block_stmts) = block.stmts.split_at(start_eq);
        let (block_stmts, end_stmts) = block_stmts.split_at(block_stmts.len() - end_eq);

        // Scan start
        let mut start_walker = UsedValueFinderVisitor::new(cx);
        for stmt in start_stmts {
            intravisit::walk_stmt(&mut start_walker, stmt);
        }
        let mut moved_syms = start_walker.def_symbols;

        // Scan block
        let mut block_walker = UsedValueFinderVisitor::new(cx);
        for stmt in block_stmts {
            intravisit::walk_stmt(&mut block_walker, stmt);
        }
        let mut block_defs = block_walker.defs;

        // Scan moved stmts
        let mut moved_start: Option<usize> = None;
        let mut end_walker = UsedValueFinderVisitor::new(cx);
        for (index, stmt) in end_stmts.iter().enumerate() {
            intravisit::walk_stmt(&mut end_walker, stmt);

            for value in &end_walker.uses {
                // Well we can't move this and all prev statements. So reset
                if block_defs.contains(value) {
                    moved_start = Some(index + 1);
                    end_walker.defs.drain().for_each(|x| {
                        block_defs.insert(x);
                    });

                    end_walker.def_symbols.clear();
                }
            }

            end_walker.uses.clear();
        }

        if let Some(moved_start) = moved_start {
            end_eq -= moved_start;
        }

        let end_linable = block.expr.map_or_else(
            || end_eq != 0,
            |expr| {
                intravisit::walk_expr(&mut end_walker, expr);
                end_walker.uses.iter().any(|x| !block_defs.contains(x))
            },
        );

        if end_linable {
            end_walker.def_symbols.drain().for_each(|x| {
                moved_syms.insert(x);
            });
        }

        emit_branches_sharing_code_lint(
            cx,
            start_eq,
            end_eq,
            end_linable,
            check_for_warn_of_moved_symbol(cx, &moved_syms, expr),
            blocks,
            expr,
        );
    }
}

struct BlockEqual {
    /// The amount statements that are equal from the start
    start_eq: usize,
    /// The amount statements that are equal from the end
    end_eq: usize,
    ///  An indication if the block expressions are the same. This will also be true if both are
    /// `None`
    expr_eq: bool,
}

/// This function can also trigger the `IF_SAME_THEN_ELSE` in which case it'll return `None` to
/// abort any further processing and avoid duplicate lint triggers.
fn scan_block_for_eq(cx: &LateContext<'tcx>, blocks: &[&Block<'tcx>]) -> Option<BlockEqual> {
    let mut start_eq = usize::MAX;
    let mut end_eq = usize::MAX;
    let mut expr_eq = true;
    let mut iter = blocks.windows(2);
    while let Some(&[win0, win1]) = iter.next() {
        let l_stmts = win0.stmts;
        let r_stmts = win1.stmts;

        // `SpanlessEq` now keeps track of the locals and is therefore context sensitive clippy#6752.
        // The comparison therefore needs to be done in a way that builds the correct context.
        let mut evaluator = SpanlessEq::new(cx);
        let mut evaluator = evaluator.inter_expr();

        let current_start_eq = count_eq(&mut l_stmts.iter(), &mut r_stmts.iter(), |l, r| evaluator.eq_stmt(l, r));

        let current_end_eq = {
            // We skip the middle statements which can't be equal
            let end_comparison_count = l_stmts.len().min(r_stmts.len()) - current_start_eq;
            let it1 = l_stmts.iter().skip(l_stmts.len() - end_comparison_count);
            let it2 = r_stmts.iter().skip(r_stmts.len() - end_comparison_count);
            it1.zip(it2)
                .fold(0, |acc, (l, r)| if evaluator.eq_stmt(l, r) { acc + 1 } else { 0 })
        };
        let block_expr_eq = both(&win0.expr, &win1.expr, |l, r| evaluator.eq_expr(l, r));

        // IF_SAME_THEN_ELSE
        if_chain! {
            if block_expr_eq;
            if l_stmts.len() == r_stmts.len();
            if l_stmts.len() == current_start_eq;
            if !is_lint_allowed(cx, IF_SAME_THEN_ELSE, win0.hir_id);
            if !is_lint_allowed(cx, IF_SAME_THEN_ELSE, win1.hir_id);
            then {
                span_lint_and_note(
                    cx,
                    IF_SAME_THEN_ELSE,
                    win0.span,
                    "this `if` has identical blocks",
                    Some(win1.span),
                    "same as this",
                );

                return None;
            }
        }

        start_eq = start_eq.min(current_start_eq);
        end_eq = end_eq.min(current_end_eq);
        expr_eq &= block_expr_eq;
    }

    if !expr_eq {
        end_eq = 0;
    }

    // Check if the regions are overlapping. Set `end_eq` to prevent the overlap
    let min_block_size = blocks.iter().map(|x| x.stmts.len()).min().unwrap();
    if (start_eq + end_eq) > min_block_size {
        end_eq = min_block_size - start_eq;
    }

    Some(BlockEqual {
        start_eq,
        end_eq,
        expr_eq,
    })
}

fn check_for_warn_of_moved_symbol(
    cx: &LateContext<'tcx>,
    symbols: &FxHashSet<Symbol>,
    if_expr: &'tcx Expr<'_>,
) -> bool {
    get_enclosing_block(cx, if_expr.hir_id).map_or(false, |block| {
        let ignore_span = block.span.shrink_to_lo().to(if_expr.span);

        symbols
            .iter()
            .filter(|sym| !sym.as_str().starts_with('_'))
            .any(move |sym| {
                let mut walker = ContainsName {
                    name: *sym,
                    result: false,
                };

                // Scan block
                block
                    .stmts
                    .iter()
                    .filter(|stmt| !ignore_span.overlaps(stmt.span))
                    .for_each(|stmt| intravisit::walk_stmt(&mut walker, stmt));

                if let Some(expr) = block.expr {
                    intravisit::walk_expr(&mut walker, expr);
                }

                walker.result
            })
    })
}

fn emit_branches_sharing_code_lint(
    cx: &LateContext<'tcx>,
    start_stmts: usize,
    end_stmts: usize,
    lint_end: bool,
    warn_about_moved_symbol: bool,
    blocks: &[&Block<'tcx>],
    if_expr: &'tcx Expr<'_>,
) {
    if start_stmts == 0 && !lint_end {
        return;
    }

    // (help, span, suggestion)
    let mut suggestions: Vec<(&str, Span, String)> = vec![];
    let mut add_expr_note = false;

    // Construct suggestions
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
        suggestions.push(("start", span, suggestion.to_string()));
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
        let moved_end = block
            .expr
            .map_or_else(|| block.stmts[block.stmts.len() - 1].span, |expr| expr.span)
            .source_callsite();

        let moved_span = moved_start.to(moved_end);
        let moved_snipped = reindent_multiline(snippet(cx, moved_span, "_"), true, None);
        let indent = indent_of(cx, if_expr.span.shrink_to_hi());
        let suggestion = "}\n".to_string() + &moved_snipped;
        let suggestion = reindent_multiline(Cow::Borrowed(&suggestion), true, indent);

        let mut span = moved_start.to(span_end);
        // Improve formatting if the inner block has indention (i.e. normal Rust formatting)
        let test_span = Span::new(span.lo() - BytePos(4), span.lo(), span.ctxt(), span.parent());
        if snippet_opt(cx, test_span)
            .map(|snip| snip == "    ")
            .unwrap_or_default()
        {
            span = span.with_lo(test_span.lo());
        }

        suggestions.push(("end", span, suggestion.to_string()));
        add_expr_note = !cx.typeck_results().expr_ty(if_expr).is_unit();
    }

    let add_optional_msgs = |diag: &mut DiagnosticBuilder<'_>| {
        if add_expr_note {
            diag.note("The end suggestion probably needs some adjustments to use the expression result correctly");
        }

        if warn_about_moved_symbol {
            diag.warn("Some moved values might need to be renamed to avoid wrong references");
        }
    };

    // Emit lint
    if suggestions.len() == 1 {
        let (place_str, span, sugg) = suggestions.pop().unwrap();
        let msg = format!("all if blocks contain the same code at the {}", place_str);
        let help = format!("consider moving the {} statements out like this", place_str);
        span_lint_and_then(cx, BRANCHES_SHARING_CODE, span, msg.as_str(), |diag| {
            diag.span_suggestion(span, help.as_str(), sugg, Applicability::Unspecified);

            add_optional_msgs(diag);
        });
    } else if suggestions.len() == 2 {
        let (_, end_span, end_sugg) = suggestions.pop().unwrap();
        let (_, start_span, start_sugg) = suggestions.pop().unwrap();
        span_lint_and_then(
            cx,
            BRANCHES_SHARING_CODE,
            start_span,
            "all if blocks contain the same code at the start and the end. Here at the start",
            move |diag| {
                diag.span_note(end_span, "and here at the end");

                diag.span_suggestion(
                    start_span,
                    "consider moving the start statements out like this",
                    start_sugg,
                    Applicability::Unspecified,
                );

                diag.span_suggestion(
                    end_span,
                    "and consider moving the end statements out like this",
                    end_sugg,
                    Applicability::Unspecified,
                );

                add_optional_msgs(diag);
            },
        );
    }
}

/// This visitor collects `HirId`s and Symbols of defined symbols and `HirId`s of used values.
struct UsedValueFinderVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,

    /// The `HirId`s of defined values in the scanned statements
    defs: FxHashSet<HirId>,

    /// The Symbols of the defined symbols in the scanned statements
    def_symbols: FxHashSet<Symbol>,

    /// The `HirId`s of the used values
    uses: FxHashSet<HirId>,
}

impl<'a, 'tcx> UsedValueFinderVisitor<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>) -> Self {
        UsedValueFinderVisitor {
            cx,
            defs: FxHashSet::default(),
            def_symbols: FxHashSet::default(),
            uses: FxHashSet::default(),
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for UsedValueFinderVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::All(self.cx.tcx.hir())
    }

    fn visit_local(&mut self, l: &'tcx rustc_hir::Local<'tcx>) {
        let local_id = l.pat.hir_id;
        self.defs.insert(local_id);

        if let Some(sym) = l.pat.simple_ident() {
            self.def_symbols.insert(sym.name);
        }

        if let Some(expr) = l.init {
            intravisit::walk_expr(self, expr);
        }
    }

    fn visit_qpath(&mut self, qpath: &'tcx rustc_hir::QPath<'tcx>, id: HirId, _span: rustc_span::Span) {
        if let rustc_hir::QPath::Resolved(_, path) = *qpath {
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
