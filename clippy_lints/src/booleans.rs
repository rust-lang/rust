use rustc::lint::{LintArray, LateLintPass, LateContext, LintPass};
use rustc::hir::*;
use rustc::hir::intravisit::*;
use syntax::ast::{LitKind, DUMMY_NODE_ID};
use syntax::codemap::{DUMMY_SP, dummy_spanned};
use utils::{span_lint_and_then, in_macro, snippet_opt, SpanlessEq};

/// **What it does:** This lint checks for boolean expressions that can be written more concisely
///
/// **Why is this bad?** Readability of boolean expressions suffers from unnecesessary duplication
///
/// **Known problems:** Ignores short circuting behavior of `||` and `&&`. Ignores `|`, `&` and `^`.
///
/// **Example:** `if a && true` should be `if a` and `!(a == b)` should be `a != b`
declare_lint! {
    pub NONMINIMAL_BOOL, Allow,
    "checks for boolean expressions that can be written more concisely"
}

/// **What it does:** This lint checks for boolean expressions that contain terminals that can be eliminated
///
/// **Why is this bad?** This is most likely a logic bug
///
/// **Known problems:** Ignores short circuiting behavior
///
/// **Example:** The `b` in `if a && b || a` is unnecessary because the expression is equivalent to `if a`
declare_lint! {
    pub LOGIC_BUG, Warn,
    "checks for boolean expressions that contain terminals which can be eliminated"
}

#[derive(Copy,Clone)]
pub struct NonminimalBool;

impl LintPass for NonminimalBool {
    fn get_lints(&self) -> LintArray {
        lint_array!(NONMINIMAL_BOOL, LOGIC_BUG)
    }
}

impl LateLintPass for NonminimalBool {
    fn check_item(&mut self, cx: &LateContext, item: &Item) {
        NonminimalBoolVisitor(cx).visit_item(item)
    }
}

struct NonminimalBoolVisitor<'a, 'tcx: 'a>(&'a LateContext<'a, 'tcx>);

use quine_mc_cluskey::Bool;
struct Hir2Qmm<'a, 'tcx: 'a, 'v> {
    terminals: Vec<&'v Expr>,
    cx: &'a LateContext<'a, 'tcx>,
}

impl<'a, 'tcx, 'v> Hir2Qmm<'a, 'tcx, 'v> {
    fn extract(&mut self, op: BinOp_, a: &[&'v Expr], mut v: Vec<Bool>) -> Result<Vec<Bool>, String> {
        for a in a {
            if let ExprBinary(binop, ref lhs, ref rhs) = a.node {
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
        if !in_macro(self.cx, e.span) {
            match e.node {
                ExprUnary(UnNot, ref inner) => return Ok(Bool::Not(box self.run(inner)?)),
                ExprBinary(binop, ref lhs, ref rhs) => {
                    match binop.node {
                        BiOr => return Ok(Bool::Or(self.extract(BiOr, &[lhs, rhs], Vec::new())?)),
                        BiAnd => return Ok(Bool::And(self.extract(BiAnd, &[lhs, rhs], Vec::new())?)),
                        _ => (),
                    }
                }
                ExprLit(ref lit) => {
                    match lit.node {
                        LitKind::Bool(true) => return Ok(Bool::True),
                        LitKind::Bool(false) => return Ok(Bool::False),
                        _ => (),
                    }
                }
                _ => (),
            }
        }
        for (n, expr) in self.terminals.iter().enumerate() {
            if SpanlessEq::new(self.cx).ignore_fn().eq_expr(e, expr) {
                #[allow(cast_possible_truncation)]
                return Ok(Bool::Term(n as u8));
            }
            let negated = match e.node {
                ExprBinary(binop, ref lhs, ref rhs) => {
                    let mk_expr = |op| {
                        Expr {
                            id: DUMMY_NODE_ID,
                            span: DUMMY_SP,
                            attrs: None,
                            node: ExprBinary(dummy_spanned(op), lhs.clone(), rhs.clone()),
                        }
                    };
                    match binop.node {
                        BiEq => mk_expr(BiNe),
                        BiNe => mk_expr(BiEq),
                        BiGt => mk_expr(BiLe),
                        BiGe => mk_expr(BiLt),
                        BiLt => mk_expr(BiGe),
                        BiLe => mk_expr(BiGt),
                        _ => continue,
                    }
                }
                _ => continue,
            };
            if SpanlessEq::new(self.cx).ignore_fn().eq_expr(&negated, expr) {
                #[allow(cast_possible_truncation)]
                return Ok(Bool::Not(Box::new(Bool::Term(n as u8))));
            }
        }
        let n = self.terminals.len();
        self.terminals.push(e);
        if n < 32 {
            #[allow(cast_possible_truncation)]
            Ok(Bool::Term(n as u8))
        } else {
            Err("too many literals".to_owned())
        }
    }
}

fn suggest(cx: &LateContext, suggestion: &Bool, terminals: &[&Expr]) -> String {
    fn recurse(brackets: bool, cx: &LateContext, suggestion: &Bool, terminals: &[&Expr], mut s: String) -> String {
        use quine_mc_cluskey::Bool::*;
        let snip = |e: &Expr| snippet_opt(cx, e.span).expect("don't try to improve booleans created by macros");
        match *suggestion {
            True => {
                s.push_str("true");
                s
            }
            False => {
                s.push_str("false");
                s
            }
            Not(ref inner) => {
                match **inner {
                    And(_) | Or(_) => {
                        s.push('!');
                        recurse(true, cx, inner, terminals, s)
                    }
                    Term(n) => {
                        if let ExprBinary(binop, ref lhs, ref rhs) = terminals[n as usize].node {
                            let op = match binop.node {
                                BiEq => " != ",
                                BiNe => " == ",
                                BiLt => " >= ",
                                BiGt => " <= ",
                                BiLe => " > ",
                                BiGe => " < ",
                                _ => {
                                    s.push('!');
                                    return recurse(true, cx, inner, terminals, s);
                                }
                            };
                            s.push_str(&snip(lhs));
                            s.push_str(op);
                            s.push_str(&snip(rhs));
                            s
                        } else {
                            s.push('!');
                            recurse(false, cx, inner, terminals, s)
                        }
                    }
                    _ => {
                        s.push('!');
                        recurse(false, cx, inner, terminals, s)
                    }
                }
            }
            And(ref v) => {
                if brackets {
                    s.push('(');
                }
                if let Or(_) = v[0] {
                    s = recurse(true, cx, &v[0], terminals, s);
                } else {
                    s = recurse(false, cx, &v[0], terminals, s);
                }
                for inner in &v[1..] {
                    s.push_str(" && ");
                    if let Or(_) = *inner {
                        s = recurse(true, cx, inner, terminals, s);
                    } else {
                        s = recurse(false, cx, inner, terminals, s);
                    }
                }
                if brackets {
                    s.push(')');
                }
                s
            }
            Or(ref v) => {
                if brackets {
                    s.push('(');
                }
                s = recurse(false, cx, &v[0], terminals, s);
                for inner in &v[1..] {
                    s.push_str(" || ");
                    s = recurse(false, cx, inner, terminals, s);
                }
                if brackets {
                    s.push(')');
                }
                s
            }
            Term(n) => {
                if brackets {
                    if let ExprBinary(..) = terminals[n as usize].node {
                        s.push('(');
                    }
                }
                s.push_str(&snip(terminals[n as usize]));
                if brackets {
                    if let ExprBinary(..) = terminals[n as usize].node {
                        s.push(')');
                    }
                }
                s
            }
        }
    }
    recurse(false, cx, suggestion, terminals, String::new())
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
        }
        Or(mut v) => {
            for el in &mut v {
                *el = simple_negate(::std::mem::replace(el, True));
            }
            And(v)
        }
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
        match *b {
            True | False => stats.ops += 1,
            Not(ref inner) => {
                match **inner {
                    And(_) | Or(_) => stats.ops += 1, // brackets are also operations
                    _ => stats.negations += 1,
                }
                recurse(inner, stats);
            }
            And(ref v) | Or(ref v) => {
                stats.ops += v.len() - 1;
                for inner in v {
                    recurse(inner, stats);
                }
            }
            Term(n) => stats.terminals[n as usize] += 1,
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
            cx: self.0,
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
                    Bool::Not(_) | Bool::True | Bool::False => {}
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
                        span_lint_and_then(self.0,
                                           LOGIC_BUG,
                                           e.span,
                                           "this boolean expression contains a logic bug",
                                           |db| {
                            db.span_help(h2q.terminals[i].span,
                                         "this expression can be optimized out by applying boolean operations to the \
                                          outer expression");
                            db.span_suggestion(e.span,
                                               "it would look like the following",
                                               suggest(self.0, suggestion, &h2q.terminals));
                        });
                        // don't also lint `NONMINIMAL_BOOL`
                        return;
                    }
                    // if the number of occurrences of a terminal decreases or any of the stats
                    // decreases while none increases
                    improvement |= (stats.terminals[i] > simplified_stats.terminals[i]) ||
                                   (stats.negations > simplified_stats.negations &&
                                    stats.ops == simplified_stats.ops) ||
                                   (stats.ops > simplified_stats.ops && stats.negations == simplified_stats.negations);
                }
                if improvement {
                    improvements.push(suggestion);
                }
            }
            if !improvements.is_empty() {
                span_lint_and_then(self.0,
                                   NONMINIMAL_BOOL,
                                   e.span,
                                   "this boolean expression can be simplified",
                                   |db| {
                    for suggestion in &improvements {
                        db.span_suggestion(e.span, "try", suggest(self.0, suggestion, &h2q.terminals));
                    }
                });
            }
        }
    }
}

impl<'a, 'v, 'tcx> Visitor<'v> for NonminimalBoolVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, e: &'v Expr) {
        if in_macro(self.0, e.span) {
            return;
        }
        match e.node {
            ExprBinary(binop, _, _) if binop.node == BiOr || binop.node == BiAnd => self.bool_expr(e),
            ExprUnary(UnNot, ref inner) => {
                if self.0.tcx.node_types()[&inner.id].is_bool() {
                    self.bool_expr(e);
                } else {
                    walk_expr(self, e);
                }
            }
            _ => walk_expr(self, e),
        }
    }
}
