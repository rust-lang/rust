use crate::utils::{
    get_trait_def_id, implements_trait, in_macro, in_macro_or_desugar, match_type, paths, snippet_opt,
    span_lint_and_then, SpanlessEq,
};
use rustc::hir::intravisit::*;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_data_structures::thin_vec::ThinVec;
use rustc_errors::Applicability;
use syntax::ast::LitKind;
use syntax::source_map::{dummy_spanned, Span, DUMMY_SP};

declare_clippy_lint! {
    /// **What it does:** Checks for boolean expressions that can be written more
    /// concisely.
    ///
    /// **Why is this bad?** Readability of boolean expressions suffers from
    /// unnecessary duplication.
    ///
    /// **Known problems:** Ignores short circuiting behavior of `||` and
    /// `&&`. Ignores `|`, `&` and `^`.
    ///
    /// **Example:**
    /// ```ignore
    /// if a && true  // should be: if a
    /// if !(a == b)  // should be: if a != b
    /// ```
    pub NONMINIMAL_BOOL,
    complexity,
    "boolean expressions that can be written more concisely"
}

declare_clippy_lint! {
    /// **What it does:** Checks for boolean expressions that contain terminals that
    /// can be eliminated.
    ///
    /// **Why is this bad?** This is most likely a logic bug.
    ///
    /// **Known problems:** Ignores short circuiting behavior.
    ///
    /// **Example:**
    /// ```ignore
    /// if a && b || a { ... }
    /// ```
    /// The `b` is unnecessary, the expression is equivalent to `if a`.
    pub LOGIC_BUG,
    correctness,
    "boolean expressions that contain terminals which can be eliminated"
}

// For each pairs, both orders are considered.
const METHODS_WITH_NEGATION: [(&str, &str); 2] = [("is_some", "is_none"), ("is_err", "is_ok")];

declare_lint_pass!(NonminimalBool => [NONMINIMAL_BOOL, LOGIC_BUG]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NonminimalBool {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        _: intravisit::FnKind<'tcx>,
        _: &'tcx FnDecl,
        body: &'tcx Body,
        _: Span,
        _: HirId,
    ) {
        NonminimalBoolVisitor { cx }.visit_body(body)
    }
}

struct NonminimalBoolVisitor<'a, 'tcx> {
    cx: &'a LateContext<'a, 'tcx>,
}

use quine_mc_cluskey::Bool;
struct Hir2Qmm<'a, 'tcx, 'v> {
    terminals: Vec<&'v Expr>,
    cx: &'a LateContext<'a, 'tcx>,
}

impl<'a, 'tcx, 'v> Hir2Qmm<'a, 'tcx, 'v> {
    fn extract(&mut self, op: BinOpKind, a: &[&'v Expr], mut v: Vec<Bool>) -> Result<Vec<Bool>, String> {
        for a in a {
            if let ExprKind::Binary(binop, lhs, rhs) = &a.node {
                if binop.node == op {
                    v = self.extract(op, &[lhs, rhs], v)?;
                    continue;
                }
            }
            v.push(self.run(a)?);
        }
        Ok(v)
    }

    fn run(&mut self, e: &'v Expr) -> Result<Bool, String> {
        // prevent folding of `cfg!` macros and the like
        if !in_macro_or_desugar(e.span) {
            match &e.node {
                ExprKind::Unary(UnNot, inner) => return Ok(Bool::Not(box self.run(inner)?)),
                ExprKind::Binary(binop, lhs, rhs) => match &binop.node {
                    BinOpKind::Or => return Ok(Bool::Or(self.extract(BinOpKind::Or, &[lhs, rhs], Vec::new())?)),
                    BinOpKind::And => return Ok(Bool::And(self.extract(BinOpKind::And, &[lhs, rhs], Vec::new())?)),
                    _ => (),
                },
                ExprKind::Lit(lit) => match lit.node {
                    LitKind::Bool(true) => return Ok(Bool::True),
                    LitKind::Bool(false) => return Ok(Bool::False),
                    _ => (),
                },
                _ => (),
            }
        }
        for (n, expr) in self.terminals.iter().enumerate() {
            if SpanlessEq::new(self.cx).ignore_fn().eq_expr(e, expr) {
                #[allow(clippy::cast_possible_truncation)]
                return Ok(Bool::Term(n as u8));
            }
            let negated = match &e.node {
                ExprKind::Binary(binop, lhs, rhs) => {
                    if !implements_ord(self.cx, lhs) {
                        continue;
                    }

                    let mk_expr = |op| Expr {
                        hir_id: DUMMY_HIR_ID,
                        span: DUMMY_SP,
                        attrs: ThinVec::new(),
                        node: ExprKind::Binary(dummy_spanned(op), lhs.clone(), rhs.clone()),
                    };
                    match binop.node {
                        BinOpKind::Eq => mk_expr(BinOpKind::Ne),
                        BinOpKind::Ne => mk_expr(BinOpKind::Eq),
                        BinOpKind::Gt => mk_expr(BinOpKind::Le),
                        BinOpKind::Ge => mk_expr(BinOpKind::Lt),
                        BinOpKind::Lt => mk_expr(BinOpKind::Ge),
                        BinOpKind::Le => mk_expr(BinOpKind::Gt),
                        _ => continue,
                    }
                },
                _ => continue,
            };
            if SpanlessEq::new(self.cx).ignore_fn().eq_expr(&negated, expr) {
                #[allow(clippy::cast_possible_truncation)]
                return Ok(Bool::Not(Box::new(Bool::Term(n as u8))));
            }
        }
        let n = self.terminals.len();
        self.terminals.push(e);
        if n < 32 {
            #[allow(clippy::cast_possible_truncation)]
            Ok(Bool::Term(n as u8))
        } else {
            Err("too many literals".to_owned())
        }
    }
}

struct SuggestContext<'a, 'tcx, 'v> {
    terminals: &'v [&'v Expr],
    cx: &'a LateContext<'a, 'tcx>,
    output: String,
    simplified: bool,
}

impl<'a, 'tcx, 'v> SuggestContext<'a, 'tcx, 'v> {
    fn snip(&self, e: &Expr) -> Option<String> {
        snippet_opt(self.cx, e.span)
    }

    fn simplify_not(&self, expr: &Expr) -> Option<String> {
        match &expr.node {
            ExprKind::Binary(binop, lhs, rhs) => {
                if !implements_ord(self.cx, lhs) {
                    return None;
                }

                match binop.node {
                    BinOpKind::Eq => Some(" != "),
                    BinOpKind::Ne => Some(" == "),
                    BinOpKind::Lt => Some(" >= "),
                    BinOpKind::Gt => Some(" <= "),
                    BinOpKind::Le => Some(" > "),
                    BinOpKind::Ge => Some(" < "),
                    _ => None,
                }
                .and_then(|op| Some(format!("{}{}{}", self.snip(lhs)?, op, self.snip(rhs)?)))
            },
            ExprKind::MethodCall(path, _, args) if args.len() == 1 => {
                let type_of_receiver = self.cx.tables.expr_ty(&args[0]);
                if !match_type(self.cx, type_of_receiver, &paths::OPTION)
                    && !match_type(self.cx, type_of_receiver, &paths::RESULT)
                {
                    return None;
                }
                METHODS_WITH_NEGATION
                    .iter()
                    .cloned()
                    .flat_map(|(a, b)| vec![(a, b), (b, a)])
                    .find(|&(a, _)| a == path.ident.name.as_str())
                    .and_then(|(_, neg_method)| Some(format!("{}.{}()", self.snip(&args[0])?, neg_method)))
            },
            _ => None,
        }
    }

    fn recurse(&mut self, suggestion: &Bool) -> Option<()> {
        use quine_mc_cluskey::Bool::*;
        match suggestion {
            True => {
                self.output.push_str("true");
            },
            False => {
                self.output.push_str("false");
            },
            Not(inner) => match **inner {
                And(_) | Or(_) => {
                    self.output.push('!');
                    self.output.push('(');
                    self.recurse(inner);
                    self.output.push(')');
                },
                Term(n) => {
                    let terminal = self.terminals[n as usize];
                    if let Some(str) = self.simplify_not(terminal) {
                        self.simplified = true;
                        self.output.push_str(&str)
                    } else {
                        self.output.push('!');
                        let snip = self.snip(terminal)?;
                        self.output.push_str(&snip);
                    }
                },
                True | False | Not(_) => {
                    self.output.push('!');
                    self.recurse(inner)?;
                },
            },
            And(v) => {
                for (index, inner) in v.iter().enumerate() {
                    if index > 0 {
                        self.output.push_str(" && ");
                    }
                    if let Or(_) = *inner {
                        self.output.push('(');
                        self.recurse(inner);
                        self.output.push(')');
                    } else {
                        self.recurse(inner);
                    }
                }
            },
            Or(v) => {
                for (index, inner) in v.iter().enumerate() {
                    if index > 0 {
                        self.output.push_str(" || ");
                    }
                    self.recurse(inner);
                }
            },
            &Term(n) => {
                let snip = self.snip(self.terminals[n as usize])?;
                self.output.push_str(&snip);
            },
        }
        Some(())
    }
}

// The boolean part of the return indicates whether some simplifications have been applied.
fn suggest(cx: &LateContext<'_, '_>, suggestion: &Bool, terminals: &[&Expr]) -> (String, bool) {
    let mut suggest_context = SuggestContext {
        terminals,
        cx,
        output: String::new(),
        simplified: false,
    };
    suggest_context.recurse(suggestion);
    (suggest_context.output, suggest_context.simplified)
}

fn simple_negate(b: Bool) -> Bool {
    use quine_mc_cluskey::Bool::*;
    match b {
        True => False,
        False => True,
        t @ Term(_) => Not(Box::new(t)),
        And(mut v) => {
            for el in &mut v {
                *el = simple_negate(::std::mem::replace(el, True));
            }
            Or(v)
        },
        Or(mut v) => {
            for el in &mut v {
                *el = simple_negate(::std::mem::replace(el, True));
            }
            And(v)
        },
        Not(inner) => *inner,
    }
}

#[derive(Default)]
struct Stats {
    terminals: [usize; 32],
    negations: usize,
    ops: usize,
}

fn terminal_stats(b: &Bool) -> Stats {
    fn recurse(b: &Bool, stats: &mut Stats) {
        match b {
            True | False => stats.ops += 1,
            Not(inner) => {
                match **inner {
                    And(_) | Or(_) => stats.ops += 1, // brackets are also operations
                    _ => stats.negations += 1,
                }
                recurse(inner, stats);
            },
            And(v) | Or(v) => {
                stats.ops += v.len() - 1;
                for inner in v {
                    recurse(inner, stats);
                }
            },
            &Term(n) => stats.terminals[n as usize] += 1,
        }
    }
    use quine_mc_cluskey::Bool::*;
    let mut stats = Stats::default();
    recurse(b, &mut stats);
    stats
}

impl<'a, 'tcx> NonminimalBoolVisitor<'a, 'tcx> {
    fn bool_expr(&self, e: &Expr) {
        let mut h2q = Hir2Qmm {
            terminals: Vec::new(),
            cx: self.cx,
        };
        if let Ok(expr) = h2q.run(e) {
            if h2q.terminals.len() > 8 {
                // QMC has exponentially slow behavior as the number of terminals increases
                // 8 is reasonable, it takes approximately 0.2 seconds.
                // See #825
                return;
            }

            let stats = terminal_stats(&expr);
            let mut simplified = expr.simplify();
            for simple in Bool::Not(Box::new(expr.clone())).simplify() {
                match simple {
                    Bool::Not(_) | Bool::True | Bool::False => {},
                    _ => simplified.push(Bool::Not(Box::new(simple.clone()))),
                }
                let simple_negated = simple_negate(simple);
                if simplified.iter().any(|s| *s == simple_negated) {
                    continue;
                }
                simplified.push(simple_negated);
            }
            let mut improvements = Vec::new();
            'simplified: for suggestion in &simplified {
                let simplified_stats = terminal_stats(suggestion);
                let mut improvement = false;
                for i in 0..32 {
                    // ignore any "simplifications" that end up requiring a terminal more often
                    // than in the original expression
                    if stats.terminals[i] < simplified_stats.terminals[i] {
                        continue 'simplified;
                    }
                    if stats.terminals[i] != 0 && simplified_stats.terminals[i] == 0 {
                        span_lint_and_then(
                            self.cx,
                            LOGIC_BUG,
                            e.span,
                            "this boolean expression contains a logic bug",
                            |db| {
                                db.span_help(
                                    h2q.terminals[i].span,
                                    "this expression can be optimized out by applying boolean operations to the \
                                     outer expression",
                                );
                                db.span_suggestion(
                                    e.span,
                                    "it would look like the following",
                                    suggest(self.cx, suggestion, &h2q.terminals).0,
                                    // nonminimal_bool can produce minimal but
                                    // not human readable expressions (#3141)
                                    Applicability::Unspecified,
                                );
                            },
                        );
                        // don't also lint `NONMINIMAL_BOOL`
                        return;
                    }
                    // if the number of occurrences of a terminal decreases or any of the stats
                    // decreases while none increases
                    improvement |= (stats.terminals[i] > simplified_stats.terminals[i])
                        || (stats.negations > simplified_stats.negations && stats.ops == simplified_stats.ops)
                        || (stats.ops > simplified_stats.ops && stats.negations == simplified_stats.negations);
                }
                if improvement {
                    improvements.push(suggestion);
                }
            }
            let nonminimal_bool_lint = |suggestions: Vec<_>| {
                span_lint_and_then(
                    self.cx,
                    NONMINIMAL_BOOL,
                    e.span,
                    "this boolean expression can be simplified",
                    |db| {
                        db.span_suggestions(
                            e.span,
                            "try",
                            suggestions.into_iter(),
                            // nonminimal_bool can produce minimal but
                            // not human readable expressions (#3141)
                            Applicability::Unspecified,
                        );
                    },
                );
            };
            if improvements.is_empty() {
                let suggest = suggest(self.cx, &expr, &h2q.terminals);
                if suggest.1 {
                    nonminimal_bool_lint(vec![suggest.0])
                }
            } else {
                nonminimal_bool_lint(
                    improvements
                        .into_iter()
                        .map(|suggestion| suggest(self.cx, suggestion, &h2q.terminals).0)
                        .collect(),
                );
            }
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for NonminimalBoolVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, e: &'tcx Expr) {
        if in_macro(e.span) {
            return;
        }
        match &e.node {
            ExprKind::Binary(binop, _, _) if binop.node == BinOpKind::Or || binop.node == BinOpKind::And => {
                self.bool_expr(e)
            },
            ExprKind::Unary(UnNot, inner) => {
                if self.cx.tables.node_types()[inner.hir_id].is_bool() {
                    self.bool_expr(e);
                } else {
                    walk_expr(self, e);
                }
            },
            _ => walk_expr(self, e),
        }
    }
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}

fn implements_ord<'a, 'tcx>(cx: &'a LateContext<'a, 'tcx>, expr: &Expr) -> bool {
    let ty = cx.tables.expr_ty(expr);
    get_trait_def_id(cx, &paths::ORD).map_or(false, |id| implements_trait(cx, ty, id, &[]))
}
