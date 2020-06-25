use crate::utils::span_lint;
use rustc_ast::ast::{FloatTy, LitFloatType, LitKind};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol;
use std::f64::consts as f64;

declare_clippy_lint! {
    /// **What it does:** Checks for floating point literals that approximate
    /// constants which are defined in
    /// [`std::f32::consts`](https://doc.rust-lang.org/stable/std/f32/consts/#constants)
    /// or
    /// [`std::f64::consts`](https://doc.rust-lang.org/stable/std/f64/consts/#constants),
    /// respectively, suggesting to use the predefined constant.
    ///
    /// **Why is this bad?** Usually, the definition in the standard library is more
    /// precise than what people come up with. If you find that your definition is
    /// actually more precise, please [file a Rust
    /// issue](https://github.com/rust-lang/rust/issues).
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// let x = 3.14;
    /// let y = 1_f64 / x;
    /// ```
    /// Use predefined constants instead:
    /// ```rust
    /// let x = std::f32::consts::PI;
    /// let y = std::f64::consts::FRAC_1_PI;
    /// ```
    pub APPROX_CONSTANT,
    correctness,
    "the approximate of a known float constant (in `std::fXX::consts`)"
}

// Tuples are of the form (constant, name, min_digits)
const KNOWN_CONSTS: [(f64, &str, usize); 18] = [
    (f64::E, "E", 4),
    (f64::FRAC_1_PI, "FRAC_1_PI", 4),
    (f64::FRAC_1_SQRT_2, "FRAC_1_SQRT_2", 5),
    (f64::FRAC_2_PI, "FRAC_2_PI", 5),
    (f64::FRAC_2_SQRT_PI, "FRAC_2_SQRT_PI", 5),
    (f64::FRAC_PI_2, "FRAC_PI_2", 5),
    (f64::FRAC_PI_3, "FRAC_PI_3", 5),
    (f64::FRAC_PI_4, "FRAC_PI_4", 5),
    (f64::FRAC_PI_6, "FRAC_PI_6", 5),
    (f64::FRAC_PI_8, "FRAC_PI_8", 5),
    (f64::LN_10, "LN_10", 5),
    (f64::LN_2, "LN_2", 5),
    (f64::LOG10_E, "LOG10_E", 5),
    (f64::LOG2_E, "LOG2_E", 5),
    (f64::LOG2_10, "LOG2_10", 5),
    (f64::LOG10_2, "LOG10_2", 5),
    (f64::PI, "PI", 3),
    (f64::SQRT_2, "SQRT_2", 5),
];

declare_lint_pass!(ApproxConstant => [APPROX_CONSTANT]);

impl<'tcx> LateLintPass<'tcx> for ApproxConstant {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let ExprKind::Lit(lit) = &e.kind {
            check_lit(cx, &lit.node, e);
        }
    }
}

fn check_lit(cx: &LateContext<'_>, lit: &LitKind, e: &Expr<'_>) {
    match *lit {
        LitKind::Float(s, LitFloatType::Suffixed(fty)) => match fty {
            FloatTy::F32 => check_known_consts(cx, e, s, "f32"),
            FloatTy::F64 => check_known_consts(cx, e, s, "f64"),
        },
        LitKind::Float(s, LitFloatType::Unsuffixed) => check_known_consts(cx, e, s, "f{32, 64}"),
        _ => (),
    }
}

fn check_known_consts(cx: &LateContext<'_>, e: &Expr<'_>, s: symbol::Symbol, module: &str) {
    let s = s.as_str();
    if s.parse::<f64>().is_ok() {
        for &(constant, name, min_digits) in &KNOWN_CONSTS {
            if is_approx_const(constant, &s, min_digits) {
                span_lint(
                    cx,
                    APPROX_CONSTANT,
                    e.span,
                    &format!(
                        "approximate value of `{}::consts::{}` found. \
                         Consider using it directly",
                        module, &name
                    ),
                );
                return;
            }
        }
    }
}

/// Returns `false` if the number of significant figures in `value` are
/// less than `min_digits`; otherwise, returns true if `value` is equal
/// to `constant`, rounded to the number of digits present in `value`.
#[must_use]
fn is_approx_const(constant: f64, value: &str, min_digits: usize) -> bool {
    if value.len() <= min_digits {
        false
    } else if constant.to_string().starts_with(value) {
        // The value is a truncated constant
        true
    } else {
        let round_const = format!("{:.*}", value.len() - 2, constant);
        value == round_const
    }
}
