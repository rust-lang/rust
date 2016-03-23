use rustc::lint::*;
use rustc_front::hir::*;
use rustc_front::intravisit::*;
use syntax::ast::LitKind;
use utils::{span_lint_and_then, in_macro, snippet_opt};

/// **What it does:** This lint checks for boolean expressions that can be written more concisely
///
/// **Why is this bad?** Readability of boolean expressions suffers from unnecesessary duplication
///
/// **Known problems:** None
///
/// **Example:** `if a && b || a` should be `if a`
declare_lint! {
    pub NONMINIMAL_BOOL, Warn,
    "checks for boolean expressions that can be written more concisely"
}

#[derive(Copy,Clone)]
pub struct NonminimalBool;

impl LintPass for NonminimalBool {
    fn get_lints(&self) -> LintArray {
        lint_array!(NONMINIMAL_BOOL)
    }
}

impl LateLintPass for NonminimalBool {
    fn check_crate(&mut self, cx: &LateContext, krate: &Crate) {
        krate.visit_all_items(&mut NonminimalBoolVisitor(cx))
    }
}

struct NonminimalBoolVisitor<'a, 'tcx: 'a>(&'a LateContext<'a, 'tcx>);

use quine_mc_cluskey::Bool;
struct Hir2Qmm<'tcx>(Vec<&'tcx Expr>);

impl<'tcx> Hir2Qmm<'tcx> {
    fn extract(&mut self, op: BinOp_, a: &[&'tcx Expr], mut v: Vec<Bool>) -> Result<Vec<Bool>, String> {
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

    fn run(&mut self, e: &'tcx Expr) -> Result<Bool, String> {
        match e.node {
            ExprUnary(UnNot, ref inner) => return Ok(Bool::Not(box self.run(inner)?)),
            ExprBinary(binop, ref lhs, ref rhs) => {
                match binop.node {
                    BiOr => return Ok(Bool::Or(self.extract(BiOr, &[lhs, rhs], Vec::new())?)),
                    BiAnd => return Ok(Bool::And(self.extract(BiAnd, &[lhs, rhs], Vec::new())?)),
                    _ => {},
                }
            },
            ExprLit(ref lit) => {
                match lit.node {
                    LitKind::Bool(true) => return Ok(Bool::True),
                    LitKind::Bool(false) => return Ok(Bool::False),
                    _ => {},
                }
            },
            _ => {},
        }
        let n = self.0.len();
        self.0.push(e);
        if n < 32 {
            #[allow(cast_possible_truncation)]
            Ok(Bool::Term(n as u8))
        } else {
            Err("too many literals".to_owned())
        }
    }
}

fn suggest(cx: &LateContext, suggestion: &Bool, terminals: &[&Expr]) -> String {
    fn recurse(cx: &LateContext, suggestion: &Bool, terminals: &[&Expr], mut s: String) -> String {
        use quine_mc_cluskey::Bool::*;
        match *suggestion {
            True => {
                s.extend("true".chars());
                s
            },
            False => {
                s.extend("false".chars());
                s
            },
            Not(ref inner) => {
                s.push('!');
                recurse(cx, inner, terminals, s)
            },
            And(ref v) => {
                s = recurse(cx, &v[0], terminals, s);
                for inner in &v[1..] {
                    s.extend(" && ".chars());
                    s = recurse(cx, inner, terminals, s);
                }
                s
            },
            Or(ref v) => {
                s = recurse(cx, &v[0], terminals, s);
                for inner in &v[1..] {
                    s.extend(" || ".chars());
                    s = recurse(cx, inner, terminals, s);
                }
                s
            },
            Term(n) => {
                s.extend(snippet_opt(cx, terminals[n as usize].span).expect("don't try to improve booleans created by macros").chars());
                s
            }
        }
    }
    recurse(cx, suggestion, terminals, String::new())
}

impl<'a, 'tcx> NonminimalBoolVisitor<'a, 'tcx> {
    fn bool_expr(&self, e: &Expr) {
        let mut h2q = Hir2Qmm(Vec::new());
        if let Ok(expr) = h2q.run(e) {
            let simplified = expr.simplify();
            if !simplified.iter().any(|s| *s == expr) {
                span_lint_and_then(self.0, NONMINIMAL_BOOL, e.span, "this boolean expression can be simplified", |db| {
                    for suggestion in &simplified {
                        db.span_suggestion(e.span, "try", suggest(self.0, suggestion, &h2q.0));
                    }
                });
            }
        }
    }
}

impl<'a, 'v, 'tcx> Visitor<'v> for NonminimalBoolVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, e: &'v Expr) {
        if in_macro(self.0, e.span) { return }
        match e.node {
            ExprBinary(binop, _, _) if binop.node == BiOr || binop.node == BiAnd => self.bool_expr(e),
            ExprUnary(UnNot, ref inner) => {
                if self.0.tcx.node_types()[&inner.id].is_bool() {
                    self.bool_expr(e);
                } else {
                    walk_expr(self, e);
                }
            },
            _ => walk_expr(self, e),
        }
    }
}
