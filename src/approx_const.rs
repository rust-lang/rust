use rustc::lint::*;
use rustc_front::hir::*;
use std::f64::consts as f64;
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

const KNOWN_CONSTS : &'static [(f64, &'static str)] = &[(f64::E, "E"), (f64::FRAC_1_PI, "FRAC_1_PI"),
    (f64::FRAC_1_SQRT_2, "FRAC_1_SQRT_2"), (f64::FRAC_2_PI, "FRAC_2_PI"),
    (f64::FRAC_2_SQRT_PI, "FRAC_2_SQRT_PI"), (f64::FRAC_PI_2, "FRAC_PI_2"), (f64::FRAC_PI_3, "FRAC_PI_3"),
    (f64::FRAC_PI_4, "FRAC_PI_4"), (f64::FRAC_PI_6, "FRAC_PI_6"), (f64::FRAC_PI_8, "FRAC_PI_8"),
    (f64::LN_10, "LN_10"), (f64::LN_2, "LN_2"), (f64::LOG10_E, "LOG10_E"), (f64::LOG2_E, "LOG2_E"),
    (f64::PI, "PI"), (f64::SQRT_2, "SQRT_2")];

const EPSILON_DIVISOR : f64 = 8192f64; //TODO: test to find a good value

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

fn check_known_consts(cx: &LateContext, e: &Expr, str: &str, module: &str) {
    if let Ok(value) = str.parse::<f64>() {
        for &(constant, name) in KNOWN_CONSTS {
            if !within_epsilon(constant, value) { continue; }
            span_lint(cx, APPROX_CONSTANT, e.span, &format!(
                "approximate value of `{}::{}` found. \
                Consider using it directly", module, &name));
        }
    }
}

fn within_epsilon(target: f64, value: f64) -> bool {
    f64::abs(value - target) < f64::abs(if target > value {
                                            target
                                        } else { value }) / EPSILON_DIVISOR
}
