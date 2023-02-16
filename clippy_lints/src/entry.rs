use clippy_utils::higher;
use clippy_utils::{
    can_move_expr_to_closure_no_visit,
    diagnostics::span_lint_and_sugg,
    is_expr_final_block_expr, is_expr_used_or_unified, match_def_path, paths, peel_hir_expr_while,
    source::{reindent_multiline, snippet_indent, snippet_with_applicability, snippet_with_context},
    SpanlessEq,
};
use core::fmt::{self, Write};
use rustc_errors::Applicability;
use rustc_hir::{
    hir_id::HirIdSet,
    intravisit::{walk_expr, Visitor},
    Block, Expr, ExprKind, Guard, HirId, Let, Pat, Stmt, StmtKind, UnOp,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{Span, SyntaxContext, DUMMY_SP};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for uses of `contains_key` + `insert` on `HashMap`
    /// or `BTreeMap`.
    ///
    /// ### Why is this bad?
    /// Using `entry` is more efficient.
    ///
    /// ### Known problems
    /// The suggestion may have type inference errors in some cases. e.g.
    /// ```rust
    /// let mut map = std::collections::HashMap::new();
    /// let _ = if !map.contains_key(&0) {
    ///     map.insert(0, 0)
    /// } else {
    ///     None
    /// };
    /// ```
    ///
    /// ### Example
    /// ```rust
    /// # use std::collections::HashMap;
    /// # let mut map = HashMap::new();
    /// # let k = 1;
    /// # let v = 1;
    /// if !map.contains_key(&k) {
    ///     map.insert(k, v);
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// # use std::collections::HashMap;
    /// # let mut map = HashMap::new();
    /// # let k = 1;
    /// # let v = 1;
    /// map.entry(k).or_insert(v);
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MAP_ENTRY,
    perf,
    "use of `contains_key` followed by `insert` on a `HashMap` or `BTreeMap`"
}

declare_lint_pass!(HashMapPass => [MAP_ENTRY]);

impl<'tcx> LateLintPass<'tcx> for HashMapPass {
    #[expect(clippy::too_many_lines)]
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }

        let Some(higher::If { cond: cond_expr, then: then_expr, r#else: else_expr }) = higher::If::hir(expr) else {
            return
        };

        let Some((map_ty, contains_expr)) = try_parse_contains(cx, cond_expr) else {
            return
        };

        let Some(then_search) = find_insert_calls(cx, &contains_expr, then_expr) else {
            return
        };

        let mut app = Applicability::MachineApplicable;
        let map_str = snippet_with_context(cx, contains_expr.map.span, contains_expr.call_ctxt, "..", &mut app).0;
        let key_str = snippet_with_context(cx, contains_expr.key.span, contains_expr.call_ctxt, "..", &mut app).0;
        let sugg = if let Some(else_expr) = else_expr {
            let Some(else_search) = find_insert_calls(cx, &contains_expr, else_expr) else {
                return;
            };

            if then_search.edits.is_empty() && else_search.edits.is_empty() {
                // No insertions
                return;
            } else if then_search.edits.is_empty() || else_search.edits.is_empty() {
                // if .. { insert } else { .. } or if .. { .. } else { insert }
                let ((then_str, entry_kind), else_str) = match (else_search.edits.is_empty(), contains_expr.negated) {
                    (true, true) => (
                        then_search.snippet_vacant(cx, then_expr.span, &mut app),
                        snippet_with_applicability(cx, else_expr.span, "{ .. }", &mut app),
                    ),
                    (true, false) => (
                        then_search.snippet_occupied(cx, then_expr.span, &mut app),
                        snippet_with_applicability(cx, else_expr.span, "{ .. }", &mut app),
                    ),
                    (false, true) => (
                        else_search.snippet_occupied(cx, else_expr.span, &mut app),
                        snippet_with_applicability(cx, then_expr.span, "{ .. }", &mut app),
                    ),
                    (false, false) => (
                        else_search.snippet_vacant(cx, else_expr.span, &mut app),
                        snippet_with_applicability(cx, then_expr.span, "{ .. }", &mut app),
                    ),
                };
                format!(
                    "if let {}::{entry_kind} = {map_str}.entry({key_str}) {then_str} else {else_str}",
                    map_ty.entry_path(),
                )
            } else {
                // if .. { insert } else { insert }
                let ((then_str, then_entry), (else_str, else_entry)) = if contains_expr.negated {
                    (
                        then_search.snippet_vacant(cx, then_expr.span, &mut app),
                        else_search.snippet_occupied(cx, else_expr.span, &mut app),
                    )
                } else {
                    (
                        then_search.snippet_occupied(cx, then_expr.span, &mut app),
                        else_search.snippet_vacant(cx, else_expr.span, &mut app),
                    )
                };
                let indent_str = snippet_indent(cx, expr.span);
                let indent_str = indent_str.as_deref().unwrap_or("");
                format!(
                    "match {map_str}.entry({key_str}) {{\n{indent_str}    {entry}::{then_entry} => {}\n\
                        {indent_str}    {entry}::{else_entry} => {}\n{indent_str}}}",
                    reindent_multiline(then_str.into(), true, Some(4 + indent_str.len())),
                    reindent_multiline(else_str.into(), true, Some(4 + indent_str.len())),
                    entry = map_ty.entry_path(),
                )
            }
        } else {
            if then_search.edits.is_empty() {
                // no insertions
                return;
            }

            // if .. { insert }
            if !then_search.allow_insert_closure {
                let (body_str, entry_kind) = if contains_expr.negated {
                    then_search.snippet_vacant(cx, then_expr.span, &mut app)
                } else {
                    then_search.snippet_occupied(cx, then_expr.span, &mut app)
                };
                format!(
                    "if let {}::{entry_kind} = {map_str}.entry({key_str}) {body_str}",
                    map_ty.entry_path(),
                )
            } else if let Some(insertion) = then_search.as_single_insertion() {
                let value_str = snippet_with_context(cx, insertion.value.span, then_expr.span.ctxt(), "..", &mut app).0;
                if contains_expr.negated {
                    if insertion.value.can_have_side_effects() {
                        format!("{map_str}.entry({key_str}).or_insert_with(|| {value_str});")
                    } else {
                        format!("{map_str}.entry({key_str}).or_insert({value_str});")
                    }
                } else {
                    // TODO: suggest using `if let Some(v) = map.get_mut(k) { .. }` here.
                    // This would need to be a different lint.
                    return;
                }
            } else {
                let block_str = then_search.snippet_closure(cx, then_expr.span, &mut app);
                if contains_expr.negated {
                    format!("{map_str}.entry({key_str}).or_insert_with(|| {block_str});")
                } else {
                    // TODO: suggest using `if let Some(v) = map.get_mut(k) { .. }` here.
                    // This would need to be a different lint.
                    return;
                }
            }
        };

        span_lint_and_sugg(
            cx,
            MAP_ENTRY,
            expr.span,
            &format!("usage of `contains_key` followed by `insert` on a `{}`", map_ty.name()),
            "try this",
            sugg,
            app,
        );
    }
}

#[derive(Clone, Copy)]
enum MapType {
    Hash,
    BTree,
}
impl MapType {
    fn name(self) -> &'static str {
        match self {
            Self::Hash => "HashMap",
            Self::BTree => "BTreeMap",
        }
    }
    fn entry_path(self) -> &'static str {
        match self {
            Self::Hash => "std::collections::hash_map::Entry",
            Self::BTree => "std::collections::btree_map::Entry",
        }
    }
}

struct ContainsExpr<'tcx> {
    negated: bool,
    map: &'tcx Expr<'tcx>,
    key: &'tcx Expr<'tcx>,
    call_ctxt: SyntaxContext,
}
fn try_parse_contains<'tcx>(cx: &LateContext<'_>, expr: &'tcx Expr<'_>) -> Option<(MapType, ContainsExpr<'tcx>)> {
    let mut negated = false;
    let expr = peel_hir_expr_while(expr, |e| match e.kind {
        ExprKind::Unary(UnOp::Not, e) => {
            negated = !negated;
            Some(e)
        },
        _ => None,
    });
    match expr.kind {
        ExprKind::MethodCall(
            _,
            map,
            [
                Expr {
                    kind: ExprKind::AddrOf(_, _, key),
                    span: key_span,
                    ..
                },
            ],
            _,
        ) if key_span.ctxt() == expr.span.ctxt() => {
            let id = cx.typeck_results().type_dependent_def_id(expr.hir_id)?;
            let expr = ContainsExpr {
                negated,
                map,
                key,
                call_ctxt: expr.span.ctxt(),
            };
            if match_def_path(cx, id, &paths::BTREEMAP_CONTAINS_KEY) {
                Some((MapType::BTree, expr))
            } else if match_def_path(cx, id, &paths::HASHMAP_CONTAINS_KEY) {
                Some((MapType::Hash, expr))
            } else {
                None
            }
        },
        _ => None,
    }
}

struct InsertExpr<'tcx> {
    map: &'tcx Expr<'tcx>,
    key: &'tcx Expr<'tcx>,
    value: &'tcx Expr<'tcx>,
}
fn try_parse_insert<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<InsertExpr<'tcx>> {
    if let ExprKind::MethodCall(_, map, [key, value], _) = expr.kind {
        let id = cx.typeck_results().type_dependent_def_id(expr.hir_id)?;
        if match_def_path(cx, id, &paths::BTREEMAP_INSERT) || match_def_path(cx, id, &paths::HASHMAP_INSERT) {
            Some(InsertExpr { map, key, value })
        } else {
            None
        }
    } else {
        None
    }
}

/// An edit that will need to be made to move the expression to use the entry api
#[derive(Clone, Copy)]
enum Edit<'tcx> {
    /// A semicolon that needs to be removed. Used to create a closure for `insert_with`.
    RemoveSemi(Span),
    /// An insertion into the map.
    Insertion(Insertion<'tcx>),
}
impl<'tcx> Edit<'tcx> {
    fn as_insertion(self) -> Option<Insertion<'tcx>> {
        if let Self::Insertion(i) = self { Some(i) } else { None }
    }
}
#[derive(Clone, Copy)]
struct Insertion<'tcx> {
    call: &'tcx Expr<'tcx>,
    value: &'tcx Expr<'tcx>,
}

/// This visitor needs to do a multiple things:
/// * Find all usages of the map. An insertion can only be made before any other usages of the map.
/// * Determine if there's an insertion using the same key. There's no need for the entry api
///   otherwise.
/// * Determine if the final statement executed is an insertion. This is needed to use
///   `or_insert_with`.
/// * Determine if there's any sub-expression that can't be placed in a closure.
/// * Determine if there's only a single insert statement. `or_insert` can be used in this case.
#[expect(clippy::struct_excessive_bools)]
struct InsertSearcher<'cx, 'tcx> {
    cx: &'cx LateContext<'tcx>,
    /// The map expression used in the contains call.
    map: &'tcx Expr<'tcx>,
    /// The key expression used in the contains call.
    key: &'tcx Expr<'tcx>,
    /// The context of the top level block. All insert calls must be in the same context.
    ctxt: SyntaxContext,
    /// Whether this expression can be safely moved into a closure.
    allow_insert_closure: bool,
    /// Whether this expression can use the entry api.
    can_use_entry: bool,
    /// Whether this expression is the final expression in this code path. This may be a statement.
    in_tail_pos: bool,
    // Is this expression a single insert. A slightly better suggestion can be made in this case.
    is_single_insert: bool,
    /// If the visitor has seen the map being used.
    is_map_used: bool,
    /// The locations where changes need to be made for the suggestion.
    edits: Vec<Edit<'tcx>>,
    /// A stack of loops the visitor is currently in.
    loops: Vec<HirId>,
    /// Local variables created in the expression. These don't need to be captured.
    locals: HirIdSet,
}
impl<'tcx> InsertSearcher<'_, 'tcx> {
    /// Visit the expression as a branch in control flow. Multiple insert calls can be used, but
    /// only if they are on separate code paths. This will return whether the map was used in the
    /// given expression.
    fn visit_cond_arm(&mut self, e: &'tcx Expr<'_>) -> bool {
        let is_map_used = self.is_map_used;
        let in_tail_pos = self.in_tail_pos;
        self.visit_expr(e);
        let res = self.is_map_used;
        self.is_map_used = is_map_used;
        self.in_tail_pos = in_tail_pos;
        res
    }

    /// Visits an expression which is not itself in a tail position, but other sibling expressions
    /// may be. e.g. if conditions
    fn visit_non_tail_expr(&mut self, e: &'tcx Expr<'_>) {
        let in_tail_pos = self.in_tail_pos;
        self.in_tail_pos = false;
        self.visit_expr(e);
        self.in_tail_pos = in_tail_pos;
    }
}
impl<'tcx> Visitor<'tcx> for InsertSearcher<'_, 'tcx> {
    fn visit_stmt(&mut self, stmt: &'tcx Stmt<'_>) {
        match stmt.kind {
            StmtKind::Semi(e) => {
                self.visit_expr(e);

                if self.in_tail_pos && self.allow_insert_closure {
                    // The spans are used to slice the top level expression into multiple parts. This requires that
                    // they all come from the same part of the source code.
                    if stmt.span.ctxt() == self.ctxt && e.span.ctxt() == self.ctxt {
                        self.edits
                            .push(Edit::RemoveSemi(stmt.span.trim_start(e.span).unwrap_or(DUMMY_SP)));
                    } else {
                        self.allow_insert_closure = false;
                    }
                }
            },
            StmtKind::Expr(e) => self.visit_expr(e),
            StmtKind::Local(l) => {
                self.visit_pat(l.pat);
                if let Some(e) = l.init {
                    self.allow_insert_closure &= !self.in_tail_pos;
                    self.in_tail_pos = false;
                    self.is_single_insert = false;
                    self.visit_expr(e);
                }
            },
            StmtKind::Item(_) => {
                self.allow_insert_closure &= !self.in_tail_pos;
                self.is_single_insert = false;
            },
        }
    }

    fn visit_block(&mut self, block: &'tcx Block<'_>) {
        // If the block is in a tail position, then the last expression (possibly a statement) is in the
        // tail position. The rest, however, are not.
        match (block.stmts, block.expr) {
            ([], None) => {
                self.allow_insert_closure &= !self.in_tail_pos;
            },
            ([], Some(expr)) => self.visit_expr(expr),
            (stmts, Some(expr)) => {
                let in_tail_pos = self.in_tail_pos;
                self.in_tail_pos = false;
                for stmt in stmts {
                    self.visit_stmt(stmt);
                }
                self.in_tail_pos = in_tail_pos;
                self.visit_expr(expr);
            },
            ([stmts @ .., stmt], None) => {
                let in_tail_pos = self.in_tail_pos;
                self.in_tail_pos = false;
                for stmt in stmts {
                    self.visit_stmt(stmt);
                }
                self.in_tail_pos = in_tail_pos;
                self.visit_stmt(stmt);
            },
        }
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if !self.can_use_entry {
            return;
        }

        match try_parse_insert(self.cx, expr) {
            Some(insert_expr) if SpanlessEq::new(self.cx).eq_expr(self.map, insert_expr.map) => {
                // Multiple inserts, inserts with a different key, and inserts from a macro can't use the entry api.
                if self.is_map_used
                    || !SpanlessEq::new(self.cx).eq_expr(self.key, insert_expr.key)
                    || expr.span.ctxt() != self.ctxt
                {
                    self.can_use_entry = false;
                    return;
                }

                self.edits.push(Edit::Insertion(Insertion {
                    call: expr,
                    value: insert_expr.value,
                }));
                self.is_map_used = true;
                self.allow_insert_closure &= self.in_tail_pos;

                // The value doesn't affect whether there is only a single insert expression.
                let is_single_insert = self.is_single_insert;
                self.visit_non_tail_expr(insert_expr.value);
                self.is_single_insert = is_single_insert;
            },
            _ if SpanlessEq::new(self.cx).eq_expr(self.map, expr) => {
                self.is_map_used = true;
            },
            _ => match expr.kind {
                ExprKind::If(cond_expr, then_expr, Some(else_expr)) => {
                    self.is_single_insert = false;
                    self.visit_non_tail_expr(cond_expr);
                    // Each branch may contain it's own insert expression.
                    let mut is_map_used = self.visit_cond_arm(then_expr);
                    is_map_used |= self.visit_cond_arm(else_expr);
                    self.is_map_used = is_map_used;
                },
                ExprKind::Match(scrutinee_expr, arms, _) => {
                    self.is_single_insert = false;
                    self.visit_non_tail_expr(scrutinee_expr);
                    // Each branch may contain it's own insert expression.
                    let mut is_map_used = self.is_map_used;
                    for arm in arms {
                        self.visit_pat(arm.pat);
                        if let Some(Guard::If(guard) | Guard::IfLet(&Let { init: guard, .. })) = arm.guard {
                            self.visit_non_tail_expr(guard);
                        }
                        is_map_used |= self.visit_cond_arm(arm.body);
                    }
                    self.is_map_used = is_map_used;
                },
                ExprKind::Loop(block, ..) => {
                    self.loops.push(expr.hir_id);
                    self.is_single_insert = false;
                    self.allow_insert_closure &= !self.in_tail_pos;
                    // Don't allow insertions inside of a loop.
                    let edit_len = self.edits.len();
                    self.visit_block(block);
                    if self.edits.len() != edit_len {
                        self.can_use_entry = false;
                    }
                    self.loops.pop();
                },
                ExprKind::Block(block, _) => self.visit_block(block),
                ExprKind::InlineAsm(_) => {
                    self.can_use_entry = false;
                },
                _ => {
                    self.allow_insert_closure &= !self.in_tail_pos;
                    self.allow_insert_closure &=
                        can_move_expr_to_closure_no_visit(self.cx, expr, &self.loops, &self.locals);
                    // Sub expressions are no longer in the tail position.
                    self.is_single_insert = false;
                    self.in_tail_pos = false;
                    walk_expr(self, expr);
                },
            },
        }
    }

    fn visit_pat(&mut self, p: &'tcx Pat<'tcx>) {
        p.each_binding_or_first(&mut |_, id, _, _| {
            self.locals.insert(id);
        });
    }
}

struct InsertSearchResults<'tcx> {
    edits: Vec<Edit<'tcx>>,
    allow_insert_closure: bool,
    is_single_insert: bool,
}
impl<'tcx> InsertSearchResults<'tcx> {
    fn as_single_insertion(&self) -> Option<Insertion<'tcx>> {
        self.is_single_insert.then(|| self.edits[0].as_insertion().unwrap())
    }

    fn snippet(
        &self,
        cx: &LateContext<'_>,
        mut span: Span,
        app: &mut Applicability,
        write_wrapped: impl Fn(&mut String, Insertion<'_>, SyntaxContext, &mut Applicability),
    ) -> String {
        let ctxt = span.ctxt();
        let mut res = String::new();
        for insertion in self.edits.iter().filter_map(|e| e.as_insertion()) {
            res.push_str(&snippet_with_applicability(
                cx,
                span.until(insertion.call.span),
                "..",
                app,
            ));
            if is_expr_used_or_unified(cx.tcx, insertion.call) {
                write_wrapped(&mut res, insertion, ctxt, app);
            } else {
                let _: fmt::Result = write!(
                    res,
                    "e.insert({})",
                    snippet_with_context(cx, insertion.value.span, ctxt, "..", app).0
                );
            }
            span = span.trim_start(insertion.call.span).unwrap_or(DUMMY_SP);
        }
        res.push_str(&snippet_with_applicability(cx, span, "..", app));
        res
    }

    fn snippet_occupied(&self, cx: &LateContext<'_>, span: Span, app: &mut Applicability) -> (String, &'static str) {
        (
            self.snippet(cx, span, app, |res, insertion, ctxt, app| {
                // Insertion into a map would return `Some(&mut value)`, but the entry returns `&mut value`
                let _: fmt::Result = write!(
                    res,
                    "Some(e.insert({}))",
                    snippet_with_context(cx, insertion.value.span, ctxt, "..", app).0
                );
            }),
            "Occupied(mut e)",
        )
    }

    fn snippet_vacant(&self, cx: &LateContext<'_>, span: Span, app: &mut Applicability) -> (String, &'static str) {
        (
            self.snippet(cx, span, app, |res, insertion, ctxt, app| {
                // Insertion into a map would return `None`, but the entry returns a mutable reference.
                let _: fmt::Result = if is_expr_final_block_expr(cx.tcx, insertion.call) {
                    write!(
                        res,
                        "e.insert({});\n{}None",
                        snippet_with_context(cx, insertion.value.span, ctxt, "..", app).0,
                        snippet_indent(cx, insertion.call.span).as_deref().unwrap_or(""),
                    )
                } else {
                    write!(
                        res,
                        "{{ e.insert({}); None }}",
                        snippet_with_context(cx, insertion.value.span, ctxt, "..", app).0,
                    )
                };
            }),
            "Vacant(e)",
        )
    }

    fn snippet_closure(&self, cx: &LateContext<'_>, mut span: Span, app: &mut Applicability) -> String {
        let ctxt = span.ctxt();
        let mut res = String::new();
        for edit in &self.edits {
            match *edit {
                Edit::Insertion(insertion) => {
                    // Cut out the value from `map.insert(key, value)`
                    res.push_str(&snippet_with_applicability(
                        cx,
                        span.until(insertion.call.span),
                        "..",
                        app,
                    ));
                    res.push_str(&snippet_with_context(cx, insertion.value.span, ctxt, "..", app).0);
                    span = span.trim_start(insertion.call.span).unwrap_or(DUMMY_SP);
                },
                Edit::RemoveSemi(semi_span) => {
                    // Cut out the semicolon. This allows the value to be returned from the closure.
                    res.push_str(&snippet_with_applicability(cx, span.until(semi_span), "..", app));
                    span = span.trim_start(semi_span).unwrap_or(DUMMY_SP);
                },
            }
        }
        res.push_str(&snippet_with_applicability(cx, span, "..", app));
        res
    }
}

fn find_insert_calls<'tcx>(
    cx: &LateContext<'tcx>,
    contains_expr: &ContainsExpr<'tcx>,
    expr: &'tcx Expr<'_>,
) -> Option<InsertSearchResults<'tcx>> {
    let mut s = InsertSearcher {
        cx,
        map: contains_expr.map,
        key: contains_expr.key,
        ctxt: expr.span.ctxt(),
        edits: Vec::new(),
        is_map_used: false,
        allow_insert_closure: true,
        can_use_entry: true,
        in_tail_pos: true,
        is_single_insert: true,
        loops: Vec::new(),
        locals: HirIdSet::default(),
    };
    s.visit_expr(expr);
    let allow_insert_closure = s.allow_insert_closure;
    let is_single_insert = s.is_single_insert;
    let edits = s.edits;
    s.can_use_entry.then_some(InsertSearchResults {
        edits,
        allow_insert_closure,
        is_single_insert,
    })
}
