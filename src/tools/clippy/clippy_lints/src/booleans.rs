use clippy_config::Conf;
use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_hir_and_then};
use clippy_utils::eq_expr_value;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::SpanRangeExt;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::{implements_trait, is_type_diagnostic_item};
use rustc_ast::ast::LitKind;
use rustc_attr_parsing::RustcVersion;
use rustc_errors::Applicability;
use rustc_hir::intravisit::{FnKind, Visitor, walk_expr};
use rustc_hir::{BinOpKind, Body, Expr, ExprKind, FnDecl, UnOp};
use rustc_lint::{LateContext, LateLintPass, Level};
use rustc_session::impl_lint_pass;
use rustc_span::def_id::LocalDefId;
use rustc_span::{Span, SyntaxContext, sym};

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
const METHODS_WITH_NEGATION: [(Option<RustcVersion>, &str, &str); 3] = [
    (None, "is_some", "is_none"),
    (None, "is_err", "is_ok"),
    (Some(msrvs::IS_NONE_OR), "is_some_and", "is_none_or"),
];

pub struct NonminimalBool {
    msrv: Msrv,
}

impl NonminimalBool {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl_lint_pass!(NonminimalBool => [NONMINIMAL_BOOL, OVERLY_COMPLEX_BOOL_EXPR]);

impl<'tcx> LateLintPass<'tcx> for NonminimalBool {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        _: FnKind<'tcx>,
        _: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        _: Span,
        _: LocalDefId,
    ) {
        NonminimalBoolVisitor { cx, msrv: self.msrv }.visit_body(body);
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        match expr.kind {
            // This check the case where an element in a boolean comparison is inverted, like:
            //
            // ```
            // let a = true;
            // !a == false;
            // ```
            ExprKind::Binary(op, left, right) if matches!(op.node, BinOpKind::Eq | BinOpKind::Ne) => {
                check_inverted_bool_in_condition(cx, expr.span, op.node, left, right);
            },
            _ => {},
        }
    }
}

fn inverted_bin_op_eq_str(op: BinOpKind) -> Option<&'static str> {
    match op {
        BinOpKind::Eq => Some("!="),
        BinOpKind::Ne => Some("=="),
        _ => None,
    }
}

fn bin_op_eq_str(op: BinOpKind) -> Option<&'static str> {
    match op {
        BinOpKind::Eq => Some("=="),
        BinOpKind::Ne => Some("!="),
        _ => None,
    }
}

fn check_inverted_bool_in_condition(
    cx: &LateContext<'_>,
    expr_span: Span,
    op: BinOpKind,
    left: &Expr<'_>,
    right: &Expr<'_>,
) {
    if expr_span.from_expansion()
        || !cx.typeck_results().node_types()[left.hir_id].is_bool()
        || !cx.typeck_results().node_types()[right.hir_id].is_bool()
    {
        return;
    }

    let suggestion = match (left.kind, right.kind) {
        (ExprKind::Unary(UnOp::Not, left_sub), ExprKind::Unary(UnOp::Not, right_sub)) => {
            let Some(left) = left_sub.span.get_source_text(cx) else {
                return;
            };
            let Some(right) = right_sub.span.get_source_text(cx) else {
                return;
            };
            let Some(op) = bin_op_eq_str(op) else { return };
            format!("{left} {op} {right}")
        },
        (ExprKind::Unary(UnOp::Not, left_sub), _) => {
            let Some(left) = left_sub.span.get_source_text(cx) else {
                return;
            };
            let Some(right) = right.span.get_source_text(cx) else {
                return;
            };
            let Some(op) = inverted_bin_op_eq_str(op) else { return };
            format!("{left} {op} {right}")
        },
        (_, ExprKind::Unary(UnOp::Not, right_sub)) => {
            let Some(left) = left.span.get_source_text(cx) else {
                return;
            };
            let Some(right) = right_sub.span.get_source_text(cx) else {
                return;
            };
            let Some(op) = inverted_bin_op_eq_str(op) else { return };
            format!("{left} {op} {right}")
        },
        _ => return,
    };
    span_lint_and_sugg(
        cx,
        NONMINIMAL_BOOL,
        expr_span,
        "this boolean expression can be simplified",
        "try",
        suggestion,
        Applicability::MachineApplicable,
    );
}

fn check_simplify_not(cx: &LateContext<'_>, msrv: Msrv, expr: &Expr<'_>) {
    if let ExprKind::Unary(UnOp::Not, inner) = &expr.kind
        && !expr.span.from_expansion()
        && !inner.span.from_expansion()
        && let Some(suggestion) = simplify_not(cx, msrv, inner)
        && cx.tcx.lint_level_at_node(NONMINIMAL_BOOL, expr.hir_id).level != Level::Allow
    {
        use clippy_utils::sugg::{Sugg, has_enclosing_paren};
        let maybe_par = if let Some(sug) = Sugg::hir_opt(cx, inner) {
            match sug {
                Sugg::BinOp(..) => true,
                Sugg::MaybeParen(sug) if !has_enclosing_paren(&sug) => true,
                _ => false,
            }
        } else {
            false
        };
        let suggestion = if maybe_par {
            format!("({suggestion})")
        } else {
            suggestion
        };
        span_lint_and_sugg(
            cx,
            NONMINIMAL_BOOL,
            expr.span,
            "this boolean expression can be simplified",
            "try",
            suggestion,
            Applicability::MachineApplicable,
        );
    }
}

struct NonminimalBoolVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    msrv: Msrv,
}

use quine_mc_cluskey::Bool;
struct Hir2Qmm<'a, 'tcx, 'v> {
    terminals: Vec<&'v Expr<'v>>,
    cx: &'a LateContext<'tcx>,
}

impl<'v> Hir2Qmm<'_, '_, 'v> {
    fn extract(&mut self, op: BinOpKind, a: &[&'v Expr<'_>], mut v: Vec<Bool>) -> Result<Vec<Bool>, String> {
        for a in a {
            if let ExprKind::Binary(binop, lhs, rhs) = &a.kind
                && binop.node == op
            {
                v = self.extract(op, &[lhs, rhs], v)?;
                continue;
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

        if self.cx.typeck_results().expr_ty(e).is_never() {
            return Err("contains never type".to_owned());
        }

        for (n, expr) in self.terminals.iter().enumerate() {
            if eq_expr_value(self.cx, e, expr) {
                #[expect(clippy::cast_possible_truncation)]
                return Ok(Bool::Term(n as u8));
            }

            if let ExprKind::Binary(e_binop, e_lhs, e_rhs) = &e.kind
                && implements_ord(self.cx, e_lhs)
                && let ExprKind::Binary(expr_binop, expr_lhs, expr_rhs) = &expr.kind
                && negate(e_binop.node) == Some(expr_binop.node)
                && eq_expr_value(self.cx, e_lhs, expr_lhs)
                && eq_expr_value(self.cx, e_rhs, expr_rhs)
            {
                #[expect(clippy::cast_possible_truncation)]
                return Ok(Bool::Not(Box::new(Bool::Term(n as u8))));
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
    msrv: Msrv,
    output: String,
}

impl SuggestContext<'_, '_, '_> {
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
                    if let Some(str) = simplify_not(self.cx, self.msrv, terminal) {
                        self.output.push_str(&str);
                    } else {
                        let mut app = Applicability::MachineApplicable;
                        let snip = Sugg::hir_with_context(self.cx, terminal, SyntaxContext::root(), "", &mut app);
                        // Ignore the case If the expression is inside a macro expansion, or the default snippet is used
                        if app != Applicability::MachineApplicable {
                            return None;
                        }
                        self.output.push_str(&(!snip).to_string());
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
                self.output.push_str(
                    &self.terminals[n as usize]
                        .span
                        .source_callsite()
                        .get_source_text(self.cx)?,
                );
            },
        }
        Some(())
    }
}

fn simplify_not(cx: &LateContext<'_>, curr_msrv: Msrv, expr: &Expr<'_>) -> Option<String> {
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
                let lhs_snippet = lhs.span.get_source_text(cx)?;
                let rhs_snippet = rhs.span.get_source_text(cx)?;

                if !(lhs_snippet.starts_with('(') && lhs_snippet.ends_with(')'))
                    && let (ExprKind::Cast(..), BinOpKind::Ge) = (&lhs.kind, binop.node)
                {
                    // e.g. `(a as u64) < b`. Without the parens the `<` is
                    // interpreted as a start of generic arguments for `u64`
                    return Some(format!("({lhs_snippet}){op}{rhs_snippet}"));
                }

                Some(format!("{lhs_snippet}{op}{rhs_snippet}"))
            })
        },
        ExprKind::MethodCall(path, receiver, args, _) => {
            let type_of_receiver = cx.typeck_results().expr_ty(receiver);
            if !is_type_diagnostic_item(cx, type_of_receiver, sym::Option)
                && !is_type_diagnostic_item(cx, type_of_receiver, sym::Result)
            {
                return None;
            }
            METHODS_WITH_NEGATION
                .iter()
                .copied()
                .flat_map(|(msrv, a, b)| vec![(msrv, a, b), (msrv, b, a)])
                .find(|&(msrv, a, _)| {
                    a == path.ident.name.as_str() && msrv.is_none_or(|msrv| curr_msrv.meets(cx, msrv))
                })
                .and_then(|(_, _, neg_method)| {
                    let negated_args = args
                        .iter()
                        .map(|arg| simplify_not(cx, curr_msrv, arg))
                        .collect::<Option<Vec<_>>>()?
                        .join(", ");
                    Some(format!(
                        "{}.{neg_method}({negated_args})",
                        receiver.span.get_source_text(cx)?
                    ))
                })
        },
        ExprKind::Closure(closure) => {
            let body = cx.tcx.hir_body(closure.body);
            let params = body
                .params
                .iter()
                .map(|param| param.span.get_source_text(cx).map(|t| t.to_string()))
                .collect::<Option<Vec<_>>>()?
                .join(", ");
            let negated = simplify_not(cx, curr_msrv, body.value)?;
            Some(format!("|{params}| {negated}"))
        },
        ExprKind::Unary(UnOp::Not, expr) => expr.span.get_source_text(cx).map(|t| t.to_string()),
        _ => None,
    }
}

fn suggest(cx: &LateContext<'_>, msrv: Msrv, suggestion: &Bool, terminals: &[&Expr<'_>]) -> String {
    let mut suggest_context = SuggestContext {
        terminals,
        cx,
        msrv,
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
                *el = simple_negate(std::mem::replace(el, True));
            }
            Or(v)
        },
        Or(mut v) => {
            for el in &mut v {
                *el = simple_negate(std::mem::replace(el, True));
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

impl<'tcx> NonminimalBoolVisitor<'_, 'tcx> {
    fn bool_expr(&self, e: &'tcx Expr<'_>) {
        let mut h2q = Hir2Qmm {
            terminals: Vec::new(),
            cx: self.cx,
        };
        if let Ok(expr) = h2q.run(e) {
            let stats = terminal_stats(&expr);
            if stats.ops > 7 {
                // QMC has exponentially slow behavior as the number of ops increases.
                // See #825, #13206
                return;
            }
            let mut simplified = expr.simplify();
            for simple in Bool::Not(Box::new(expr)).simplify() {
                match simple {
                    Bool::Not(_) | Bool::True | Bool::False => {},
                    _ => simplified.push(Bool::Not(Box::new(simple.clone()))),
                }
                let simple_negated = simple_negate(simple);
                if simplified.contains(&simple_negated) {
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
                                    suggest(self.cx, self.msrv, suggestion, &h2q.terminals),
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
            let nonminimal_bool_lint = |mut suggestions: Vec<_>| {
                if self.cx.tcx.lint_level_at_node(NONMINIMAL_BOOL, e.hir_id).level != Level::Allow {
                    suggestions.sort();
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
                                suggestions,
                                // nonminimal_bool can produce minimal but
                                // not human readable expressions (#3141)
                                Applicability::Unspecified,
                            );
                        },
                    );
                }
            };
            if improvements.is_empty() {
                check_simplify_not(self.cx, self.msrv, e);
            } else {
                nonminimal_bool_lint(
                    improvements
                        .into_iter()
                        .map(|suggestion| suggest(self.cx, self.msrv, suggestion, &h2q.terminals))
                        .collect(),
                );
            }
        }
    }
}

impl<'tcx> Visitor<'tcx> for NonminimalBoolVisitor<'_, 'tcx> {
    fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
        if !e.span.from_expansion() {
            match &e.kind {
                ExprKind::Binary(binop, _, _) if binop.node == BinOpKind::Or || binop.node == BinOpKind::And => {
                    self.bool_expr(e);
                },
                ExprKind::Unary(UnOp::Not, inner) => {
                    if let ExprKind::Unary(UnOp::Not, ex) = inner.kind
                        && !self.cx.typeck_results().node_types()[ex.hir_id].is_bool()
                    {
                        return;
                    }
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
        .is_some_and(|id| implements_trait(cx, ty, id, &[]))
}
