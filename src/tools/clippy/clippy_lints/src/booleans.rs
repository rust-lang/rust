use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_hir_and_then};
use clippy_utils::eq_expr_value;
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::{implements_trait, is_type_diagnostic_item};
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::intravisit::{walk_expr, FnKind, Visitor};
use rustc_hir::{BinOpKind, Body, Expr, ExprKind, FnDecl, HirId, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for boolean expressions that can be written more
    /// concisely.
    ///
    /// ### Why is this bad?
    /// Readability of boolean expressions suffers from
    /// unnecessary duplication.
    ///
    /// ### Known problems
    /// Ignores short circuiting behavior of `||` and
    /// `&&`. Ignores `|`, `&` and `^`.
    ///
    /// ### Example
    /// ```ignore
    /// if a && true {}
    /// if !(a == b) {}
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// if a {}
    /// if a != b {}
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NONMINIMAL_BOOL,
    complexity,
    "boolean expressions that can be written more concisely"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for boolean expressions that contain terminals that
    /// can be eliminated.
    ///
    /// ### Why is this bad?
    /// This is most likely a logic bug.
    ///
    /// ### Known problems
    /// Ignores short circuiting behavior.
    ///
    /// ### Example
    /// ```rust,ignore
    /// // The `b` is unnecessary, the expression is equivalent to `if a`.
    /// if a && b || a { ... }
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// if a {}
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub OVERLY_COMPLEX_BOOL_EXPR,
    correctness,
    "boolean expressions that contain terminals which can be eliminated"
}

// For each pairs, both orders are considered.
const METHODS_WITH_NEGATION: [(&str, &str); 2] = [("is_some", "is_none"), ("is_err", "is_ok")];

declare_lint_pass!(NonminimalBool => [NONMINIMAL_BOOL, OVERLY_COMPLEX_BOOL_EXPR]);

impl<'tcx> LateLintPass<'tcx> for NonminimalBool {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        _: FnKind<'tcx>,
        _: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        _: Span,
        _: HirId,
    ) {
        NonminimalBoolVisitor { cx }.visit_body(body);
    }
}

struct NonminimalBoolVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
}

use quine_mc_cluskey::Bool;
struct Hir2Qmm<'a, 'tcx, 'v> {
    terminals: Vec<&'v Expr<'v>>,
    cx: &'a LateContext<'tcx>,
}

impl<'a, 'tcx, 'v> Hir2Qmm<'a, 'tcx, 'v> {
    fn extract(&mut self, op: BinOpKind, a: &[&'v Expr<'_>], mut v: Vec<Bool>) -> Result<Vec<Bool>, String> {
        for a in a {
            if let ExprKind::Binary(binop, lhs, rhs) = &a.kind {
                if binop.node == op {
                    v = self.extract(op, &[lhs, rhs], v)?;
                    continue;
                }
            }
            v.push(self.run(a)?);
        }
        Ok(v)
    }

    fn run(&mut self, e: &'v Expr<'_>) -> Result<Bool, String> {
        fn negate(bin_op_kind: BinOpKind) -> Option<BinOpKind> {
            match bin_op_kind {
                BinOpKind::Eq => Some(BinOpKind::Ne),
                BinOpKind::Ne => Some(BinOpKind::Eq),
                BinOpKind::Gt => Some(BinOpKind::Le),
                BinOpKind::Ge => Some(BinOpKind::Lt),
                BinOpKind::Lt => Some(BinOpKind::Ge),
                BinOpKind::Le => Some(BinOpKind::Gt),
                _ => None,
            }
        }

        // prevent folding of `cfg!` macros and the like
        if !e.span.from_expansion() {
            match &e.kind {
                ExprKind::Unary(UnOp::Not, inner) => return Ok(Bool::Not(Box::new(self.run(inner)?))),
                ExprKind::Binary(binop, lhs, rhs) => match &binop.node {
                    BinOpKind::Or => {
                        return Ok(Bool::Or(self.extract(BinOpKind::Or, &[lhs, rhs], Vec::new())?));
                    },
                    BinOpKind::And => {
                        return Ok(Bool::And(self.extract(BinOpKind::And, &[lhs, rhs], Vec::new())?));
                    },
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
            if eq_expr_value(self.cx, e, expr) {
                #[expect(clippy::cast_possible_truncation)]
                return Ok(Bool::Term(n as u8));
            }

            if_chain! {
                if let ExprKind::Binary(e_binop, e_lhs, e_rhs) = &e.kind;
                if implements_ord(self.cx, e_lhs);
                if let ExprKind::Binary(expr_binop, expr_lhs, expr_rhs) = &expr.kind;
                if negate(e_binop.node) == Some(expr_binop.node);
                if eq_expr_value(self.cx, e_lhs, expr_lhs);
                if eq_expr_value(self.cx, e_rhs, expr_rhs);
                then {
                    #[expect(clippy::cast_possible_truncation)]
                    return Ok(Bool::Not(Box::new(Bool::Term(n as u8))));
                }
            }
        }
        let n = self.terminals.len();
        self.terminals.push(e);
        if n < 32 {
            #[expect(clippy::cast_possible_truncation)]
            Ok(Bool::Term(n as u8))
        } else {
            Err("too many literals".to_owned())
        }
    }
}

struct SuggestContext<'a, 'tcx, 'v> {
    terminals: &'v [&'v Expr<'v>],
    cx: &'a LateContext<'tcx>,
    output: String,
}

impl<'a, 'tcx, 'v> SuggestContext<'a, 'tcx, 'v> {
    fn recurse(&mut self, suggestion: &Bool) -> Option<()> {
        use quine_mc_cluskey::Bool::{And, False, Not, Or, Term, True};
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
                    if let Some(str) = simplify_not(self.cx, terminal) {
                        self.output.push_str(&str);
                    } else {
                        self.output.push('!');
                        let snip = snippet_opt(self.cx, terminal.span)?;
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
                for (index, inner) in v.iter().rev().enumerate() {
                    if index > 0 {
                        self.output.push_str(" || ");
                    }
                    self.recurse(inner);
                }
            },
            &Term(n) => {
                let snip = snippet_opt(self.cx, self.terminals[n as usize].span.source_callsite())?;
                self.output.push_str(&snip);
            },
        }
        Some(())
    }
}

fn simplify_not(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<String> {
    match &expr.kind {
        ExprKind::Binary(binop, lhs, rhs) => {
            if !implements_ord(cx, lhs) {
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
            .and_then(|op| {
                Some(format!(
                    "{}{op}{}",
                    snippet_opt(cx, lhs.span)?,
                    snippet_opt(cx, rhs.span)?
                ))
            })
        },
        ExprKind::MethodCall(path, receiver, [], _) => {
            let type_of_receiver = cx.typeck_results().expr_ty(receiver);
            if !is_type_diagnostic_item(cx, type_of_receiver, sym::Option)
                && !is_type_diagnostic_item(cx, type_of_receiver, sym::Result)
            {
                return None;
            }
            METHODS_WITH_NEGATION
                .iter()
                .copied()
                .flat_map(|(a, b)| vec![(a, b), (b, a)])
                .find(|&(a, _)| {
                    let path: &str = path.ident.name.as_str();
                    a == path
                })
                .and_then(|(_, neg_method)| Some(format!("{}.{neg_method}()", snippet_opt(cx, receiver.span)?)))
        },
        _ => None,
    }
}

fn suggest(cx: &LateContext<'_>, suggestion: &Bool, terminals: &[&Expr<'_>]) -> String {
    let mut suggest_context = SuggestContext {
        terminals,
        cx,
        output: String::new(),
    };
    suggest_context.recurse(suggestion);
    suggest_context.output
}

fn simple_negate(b: Bool) -> Bool {
    use quine_mc_cluskey::Bool::{And, False, Not, Or, Term, True};
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
    use quine_mc_cluskey::Bool::{And, False, Not, Or, Term, True};
    let mut stats = Stats::default();
    recurse(b, &mut stats);
    stats
}

impl<'a, 'tcx> NonminimalBoolVisitor<'a, 'tcx> {
    fn bool_expr(&self, e: &'tcx Expr<'_>) {
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
            for simple in Bool::Not(Box::new(expr)).simplify() {
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
            let mut improvements = Vec::with_capacity(simplified.len());
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
                        span_lint_hir_and_then(
                            self.cx,
                            OVERLY_COMPLEX_BOOL_EXPR,
                            e.hir_id,
                            e.span,
                            "this boolean expression contains a logic bug",
                            |diag| {
                                diag.span_help(
                                    h2q.terminals[i].span,
                                    "this expression can be optimized out by applying boolean operations to the \
                                     outer expression",
                                );
                                diag.span_suggestion(
                                    e.span,
                                    "it would look like the following",
                                    suggest(self.cx, suggestion, &h2q.terminals),
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
                span_lint_hir_and_then(
                    self.cx,
                    NONMINIMAL_BOOL,
                    e.hir_id,
                    e.span,
                    "this boolean expression can be simplified",
                    |diag| {
                        diag.span_suggestions(
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
                let mut visitor = NotSimplificationVisitor { cx: self.cx };
                visitor.visit_expr(e);
            } else {
                nonminimal_bool_lint(
                    improvements
                        .into_iter()
                        .map(|suggestion| suggest(self.cx, suggestion, &h2q.terminals))
                        .collect(),
                );
            }
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for NonminimalBoolVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
        if !e.span.from_expansion() {
            match &e.kind {
                ExprKind::Binary(binop, _, _) if binop.node == BinOpKind::Or || binop.node == BinOpKind::And => {
                    self.bool_expr(e);
                },
                ExprKind::Unary(UnOp::Not, inner) => {
                    if self.cx.typeck_results().node_types()[inner.hir_id].is_bool() {
                        self.bool_expr(e);
                    }
                },
                _ => {},
            }
        }
        walk_expr(self, e);
    }
}

fn implements_ord(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let ty = cx.typeck_results().expr_ty(expr);
    cx.tcx
        .get_diagnostic_item(sym::Ord)
        .map_or(false, |id| implements_trait(cx, ty, id, &[]))
}

struct NotSimplificationVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for NotSimplificationVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if let ExprKind::Unary(UnOp::Not, inner) = &expr.kind {
            if let Some(suggestion) = simplify_not(self.cx, inner) {
                span_lint_and_sugg(
                    self.cx,
                    NONMINIMAL_BOOL,
                    expr.span,
                    "this boolean expression can be simplified",
                    "try",
                    suggestion,
                    Applicability::MachineApplicable,
                );
            }
        }

        walk_expr(self, expr);
    }
}
