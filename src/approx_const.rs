use rustc::lint::*;
use rustc_front::hir::*;
use utils::span_lint;
use syntax::ast::Lit_::*;
use syntax::ast::Lit;
use syntax::ast::FloatTy::*;

declare_lint! {
    pub APPROX_CONSTANT,
    Warn,
    "the approximate of a known float constant (in `std::f64::consts` or `std::f32::consts`) \
     is found; suggests to use the constant"
}

// Tuples are of the form (name, lower_bound, upper_bound)
#[allow(approx_constant)]
const KNOWN_CONSTS : &'static [(&'static str, f64, f64)] = &[
    ("E", 2.7101, 2.7200),
    ("FRAC_1_PI", 0.31829, 0.31840),
    ("FRAC_1_SQRT_2", 0.7071, 0.7072),
    ("FRAC_2_PI", 0.6366, 0.6370),
    ("FRAC_2_SQRT_PI", 1.1283, 1.1284),
    ("FRAC_PI_2", 1.5707, 1.5708),
    ("FRAC_PI_3", 1.0471, 1.0472),
    ("FRAC_PI_4", 0.7853, 0.7854),
    ("FRAC_PI_6", 0.5235, 0.5236),
    ("FRAC_PI_8", 0.3926, 0.3927),
    ("LN_10", 2.302, 2.303),
    ("LN_2", 0.6931, 0.6932),
    ("LOG10_E", 0.4342, 0.4343),
    ("LOG2_E", 1.4426, 1.4427),
    ("PI", 3.140, 3.142),
    ("SQRT_2", 1.4142, 1.4143),
];

#[derive(Copy,Clone)]
pub struct ApproxConstant;

impl LintPass for ApproxConstant {
    fn get_lints(&self) -> LintArray {
        lint_array!(APPROX_CONSTANT)
    }
}

impl LateLintPass for ApproxConstant {
    fn check_expr(&mut self, cx: &LateContext, e: &Expr) {
        if let &ExprLit(ref lit) = &e.node {
            check_lit(cx, lit, e);
        }
    }
}

fn check_lit(cx: &LateContext, lit: &Lit, e: &Expr) {
    match lit.node {
        LitFloat(ref str, TyF32) => check_known_consts(cx, e, str, "f32"),
        LitFloat(ref str, TyF64) => check_known_consts(cx, e, str, "f64"),
        LitFloatUnsuffixed(ref str) =>
            check_known_consts(cx, e, str, "f{32, 64}"),
        _ => ()
    }
}

fn check_known_consts(cx: &LateContext, e: &Expr, s: &str, module: &str) {
    if let Ok(value) = s.parse::<f64>() {
        for &(name, lower_bound, upper_bound) in KNOWN_CONSTS {
            if (value >= lower_bound) && (value < upper_bound) {
                span_lint(cx, APPROX_CONSTANT, e.span, &format!(
                    "approximate value of `{}::{}` found. \
                    Consider using it directly", module, &name));
            }
        }
    }
}
