use clippy_utils::{
    diagnostics::span_lint_and_sugg,
    is_expr_final_block_expr, is_expr_used_or_unified, match_def_path, paths, peel_hir_expr_while,
    source::{snippet_indent, snippet_with_applicability, snippet_with_context},
    SpanlessEq,
};
use rustc_errors::Applicability;
use rustc_hir::{
    intravisit::{walk_expr, ErasedMap, NestedVisitorMap, Visitor},
    Expr, ExprKind, Guard, Local, Stmt, StmtKind, UnOp,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{Span, SyntaxContext, DUMMY_SP};
use std::fmt::Write;

declare_clippy_lint! {
    /// **What it does:** Checks for uses of `contains_key` + `insert` on `HashMap`
    /// or `BTreeMap`.
    ///
    /// **Why is this bad?** Using `entry` is more efficient.
    ///
    /// **Known problems:** The suggestion may have type inference errors in some cases. e.g.
    /// ```rust
    /// let mut map = std::collections::HashMap::new();
    /// let _ = if !map.contains_key(&0) {
    ///     map.insert(0, 0)
    /// } else {
    ///     None
    /// };
    /// ```
    ///
    /// **Example:**
    /// ```rust
    /// # use std::collections::HashMap;
    /// # let mut map = HashMap::new();
    /// # let k = 1;
    /// # let v = 1;
    /// if !map.contains_key(&k) {
    ///     map.insert(k, v);
    /// }
    /// ```
    /// can both be rewritten as:
    /// ```rust
    /// # use std::collections::HashMap;
    /// # let mut map = HashMap::new();
    /// # let k = 1;
    /// # let v = 1;
    /// map.entry(k).or_insert(v);
    /// ```
    pub MAP_ENTRY,
    perf,
    "use of `contains_key` followed by `insert` on a `HashMap` or `BTreeMap`"
}

declare_lint_pass!(HashMapPass => [MAP_ENTRY]);

impl<'tcx> LateLintPass<'tcx> for HashMapPass {
    #[allow(clippy::too_many_lines)]
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let (cond_expr, then_expr, else_expr) = match expr.kind {
            ExprKind::If(c, t, e) => (c, t, e),
            _ => return,
        };
        let (map_ty, contains_expr) = match try_parse_contains(cx, cond_expr) {
            Some(x) => x,
            None => return,
        };

        let then_search = match find_insert_calls(cx, &contains_expr, then_expr) {
            Some(x) => x,
            None => return,
        };

        let mut app = Applicability::MachineApplicable;
        let map_str = snippet_with_context(cx, contains_expr.map.span, contains_expr.call_ctxt, "..", &mut app).0;
        let key_str = snippet_with_context(cx, contains_expr.key.span, contains_expr.call_ctxt, "..", &mut app).0;
        let sugg = if !contains_expr.negated || else_expr.is_some() || then_search.insertions.is_empty() {
            return;
        } else {
            // if .. { insert }
            match then_search.as_single_insertion() {
                Some(insertion) if !insertion.value.can_have_side_effects() => {
                    format!(
                        "{}.entry({}).or_insert({});",
                        map_str,
                        key_str,
                        snippet_with_context(cx, insertion.value.span, insertion.call.span.ctxt(), "..", &mut app).0,
                    )
                },
                _ => {
                    let (body_str, entry_kind) = if contains_expr.negated {
                        (then_search.snippet_vacant(cx, then_expr.span, &mut app), "Vacant(e)")
                    } else {
                        (
                            then_search.snippet_occupied(cx, then_expr.span, &mut app),
                            "Occupied(mut e)",
                        )
                    };
                    format!(
                        "if let {}::{} = {}.entry({}) {}",
                        map_ty.entry_path(),
                        entry_kind,
                        map_str,
                        key_str,
                        body_str,
                    )
                },
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
    fn entry_path(self) -> &'staic str {
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
fn try_parse_contains(cx: &LateContext<'_>, expr: &'tcx Expr<'_>) -> Option<(MapType, ContainsExpr<'tcx>)> {
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
            _,
            [map, Expr {
                kind: ExprKind::AddrOf(_, _, key),
                span: key_span,
                ..
            }],
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
fn try_parse_insert(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<InsertExpr<'tcx>> {
    if let ExprKind::MethodCall(_, _, [map, key, value], _) = expr.kind {
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

#[derive(Clone, Copy)]
struct Insertion<'tcx> {
    call: &'tcx Expr<'tcx>,
    value: &'tcx Expr<'tcx>,
}

// This visitor needs to do a multiple things:
// * Find all usages of the map. Only insertions into the map which share the same key are
//   permitted. All others will prevent the lint.
// * Determine if the final statement executed is an insertion. This is needed to use `insert_with`.
// * Determine if there's any sub-expression that can't be placed in a closure.
// * Determine if there's only a single insert statement. This is needed to give better suggestions.

#[allow(clippy::struct_excessive_bools)]
struct InsertSearcher<'cx, 'i, 'tcx> {
    cx: &'cx LateContext<'tcx>,
    /// The map expression used in the contains call.
    map: &'tcx Expr<'tcx>,
    /// The key expression used in the contains call.
    key: &'tcx Expr<'tcx>,
    /// The context of the top level block. All insert calls must be in the same context.
    ctxt: SyntaxContext,
    /// Whether this expression can use the entry api.
    can_use_entry: bool,
    // A single insert expression has a slightly different suggestion.
    is_single_insert: bool,
    is_map_used: bool,
    insertions: &'i mut Vec<Insertion<'tcx>>,
}
impl<'tcx> InsertSearcher<'_, '_, 'tcx> {
    /// Visit the expression as a branch in control flow. Multiple insert calls can be used, but
    /// only if they are on separate code paths. This will return whether the map was used in the
    /// given expression.
    fn visit_cond_arm(&mut self, e: &'tcx Expr<'_>) -> bool {
        let is_map_used = self.is_map_used;
        self.visit_expr(e);
        let res = self.is_map_used;
        self.is_map_used = is_map_used;
        res
    }
}
impl<'tcx> Visitor<'tcx> for InsertSearcher<'_, '_, 'tcx> {
    type Map = ErasedMap<'tcx>;
    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

    fn visit_stmt(&mut self, stmt: &'tcx Stmt<'_>) {
        match stmt.kind {
            StmtKind::Semi(e) | StmtKind::Expr(e) => self.visit_expr(e),
            StmtKind::Local(Local { init: Some(e), .. }) => {
                self.is_single_insert = false;
                self.visit_expr(e);
            },
            _ => {
                self.is_single_insert = false;
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

                self.insertions.push(Insertion {
                    call: expr,
                    value: insert_expr.value,
                });
                self.is_map_used = true;

                // The value doesn't affect whether there is only a single insert expression.
                let is_single_insert = self.is_single_insert;
                self.visit_expr(insert_expr.value);
                self.is_single_insert = is_single_insert;
            },
            _ if SpanlessEq::new(self.cx).eq_expr(self.map, expr) => {
                self.is_map_used = true;
            },
            _ => match expr.kind {
                ExprKind::If(cond_expr, then_expr, Some(else_expr)) => {
                    self.is_single_insert = false;
                    self.visit_expr(cond_expr);
                    // Each branch may contain it's own insert expression.
                    let mut is_map_used = self.visit_cond_arm(then_expr);
                    is_map_used |= self.visit_cond_arm(else_expr);
                    self.is_map_used = is_map_used;
                },
                ExprKind::Match(scrutinee_expr, arms, _) => {
                    self.is_single_insert = false;
                    self.visit_expr(scrutinee_expr);
                    // Each branch may contain it's own insert expression.
                    let mut is_map_used = self.is_map_used;
                    for arm in arms {
                        if let Some(Guard::If(guard) | Guard::IfLet(_, guard)) = arm.guard {
                            self.visit_expr(guard)
                        }
                        is_map_used |= self.visit_cond_arm(arm.body);
                    }
                    self.is_map_used = is_map_used;
                },
                ExprKind::Loop(block, ..) => {
                    // Don't allow insertions inside of a loop.
                    let insertions_len = self.insertions.len();
                    self.visit_block(block);
                    if self.insertions.len() != insertions_len {
                        self.can_use_entry = false;
                    }
                },
                ExprKind::Block(block, _) => self.visit_block(block),
                ExprKind::InlineAsm(_) | ExprKind::LlvmInlineAsm(_) => {
                    self.can_use_entry = false;
                },
                _ => {
                    self.is_single_insert = false;
                    walk_expr(self, expr);
                },
            },
        }
    }
}

struct InsertSearchResults<'tcx> {
    insertions: Vec<Insertion<'tcx>>,
    is_single_insert: bool,
}
impl InsertSearchResults<'tcx> {
    fn as_single_insertion(&self) -> Option<Insertion<'tcx>> {
        self.is_single_insert.then(|| self.insertions[0])
    }

    fn snippet_occupied(&self, cx: &LateContext<'_>, mut span: Span, app: &mut Applicability) -> String {
        let ctxt = span.ctxt();
        let mut res = String::new();
        for insertion in self.insertions.iter() {
            res.push_str(&snippet_with_applicability(
                cx,
                span.until(insertion.call.span),
                "..",
                app,
            ));
            if is_expr_used_or_unified(cx.tcx, insertion.call) {
                res.push_str("Some(e.insert(");
                res.push_str(&snippet_with_context(cx, insertion.value.span, ctxt, "..", app).0);
                res.push_str("))");
            } else {
                res.push_str("e.insert(");
                res.push_str(&snippet_with_context(cx, insertion.value.span, ctxt, "..", app).0);
                res.push(')');
            }
            span = span.trim_start(insertion.call.span).unwrap_or(DUMMY_SP);
        }
        res.push_str(&snippet_with_applicability(cx, span, "..", app));
        res
    }

    fn snippet_vacant(&self, cx: &LateContext<'_>, mut span: Span, app: &mut Applicability) -> String {
        let ctxt = span.ctxt();
        let mut res = String::new();
        for insertion in self.insertions.iter() {
            res.push_str(&snippet_with_applicability(
                cx,
                span.until(insertion.call.span),
                "..",
                app,
            ));
            if is_expr_used_or_unified(cx.tcx, insertion.call) {
                if is_expr_final_block_expr(cx.tcx, insertion.call) {
                    let _ = write!(
                        res,
                        "e.insert({});\n{}None",
                        snippet_with_context(cx, insertion.value.span, ctxt, "..", app).0,
                        snippet_indent(cx, insertion.call.span).as_deref().unwrap_or(""),
                    );
                } else {
                    let _ = write!(
                        res,
                        "{{ e.insert({}); None }}",
                        snippet_with_context(cx, insertion.value.span, ctxt, "..", app).0,
                    );
                }
            } else {
                let _ = write!(
                    res,
                    "e.insert({})",
                    snippet_with_context(cx, insertion.value.span, ctxt, "..", app).0,
                );
            }
            span = span.trim_start(insertion.call.span).unwrap_or(DUMMY_SP);
        }
        res.push_str(&snippet_with_applicability(cx, span, "..", app));
        res
    }
}
fn find_insert_calls(
    cx: &LateContext<'tcx>,
    contains_expr: &ContainsExpr<'tcx>,
    expr: &'tcx Expr<'_>,
) -> Option<InsertSearchResults<'tcx>> {
    let mut insertions = Vec::new();
    let mut s = InsertSearcher {
        cx,
        map: contains_expr.map,
        key: contains_expr.key,
        ctxt: expr.span.ctxt(),
        insertions: &mut insertions,
        is_map_used: false,
        can_use_entry: true,
        is_single_insert: true,
    };
    s.visit_expr(expr);
    let is_single_insert = s.is_single_insert;
    s.can_use_entry.then(|| InsertSearchResults {
        insertions,
        is_single_insert,
    })
}
