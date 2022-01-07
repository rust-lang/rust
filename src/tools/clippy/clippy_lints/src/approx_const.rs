use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::{meets_msrv, msrvs};
use rustc_ast::ast::{FloatTy, LitFloatType, LitKind};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol;
use std::f64::consts as f64;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for floating point literals that approximate
    /// constants which are defined in
    /// [`std::f32::consts`](https://doc.rust-lang.org/stable/std/f32/consts/#constants)
    /// or
    /// [`std::f64::consts`](https://doc.rust-lang.org/stable/std/f64/consts/#constants),
    /// respectively, suggesting to use the predefined constant.
    ///
    /// ### Why is this bad?
    /// Usually, the definition in the standard library is more
    /// precise than what people come up with. If you find that your definition is
    /// actually more precise, please [file a Rust
    /// issue](https://github.com/rust-lang/rust/issues).
    ///
    /// ### Example
    /// ```rust
    /// let x = 3.14;
    /// let y = 1_f64 / x;
    /// ```
    /// Use predefined constants instead:
    /// ```rust
    /// let x = std::f32::consts::PI;
    /// let y = std::f64::consts::FRAC_1_PI;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub APPROX_CONSTANT,
    correctness,
    "the approximate of a known float constant (in `std::fXX::consts`)"
}

// Tuples are of the form (constant, name, min_digits, msrv)
const KNOWN_CONSTS: [(f64, &str, usize, Option<RustcVersion>); 19] = [
    (f64::E, "E", 4, None),
    (f64::FRAC_1_PI, "FRAC_1_PI", 4, None),
    (f64::FRAC_1_SQRT_2, "FRAC_1_SQRT_2", 5, None),
    (f64::FRAC_2_PI, "FRAC_2_PI", 5, None),
    (f64::FRAC_2_SQRT_PI, "FRAC_2_SQRT_PI", 5, None),
    (f64::FRAC_PI_2, "FRAC_PI_2", 5, None),
    (f64::FRAC_PI_3, "FRAC_PI_3", 5, None),
    (f64::FRAC_PI_4, "FRAC_PI_4", 5, None),
    (f64::FRAC_PI_6, "FRAC_PI_6", 5, None),
    (f64::FRAC_PI_8, "FRAC_PI_8", 5, None),
    (f64::LN_2, "LN_2", 5, None),
    (f64::LN_10, "LN_10", 5, None),
    (f64::LOG2_10, "LOG2_10", 5, Some(msrvs::LOG2_10)),
    (f64::LOG2_E, "LOG2_E", 5, None),
    (f64::LOG10_2, "LOG10_2", 5, Some(msrvs::LOG10_2)),
    (f64::LOG10_E, "LOG10_E", 5, None),
    (f64::PI, "PI", 3, None),
    (f64::SQRT_2, "SQRT_2", 5, None),
    (f64::TAU, "TAU", 3, Some(msrvs::TAU)),
];

pub struct ApproxConstant {
    msrv: Option<RustcVersion>,
}

impl ApproxConstant {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
    }

    fn check_lit(&self, cx: &LateContext<'_>, lit: &LitKind, e: &Expr<'_>) {
        match *lit {
            LitKind::Float(s, LitFloatType::Suffixed(fty)) => match fty {
                FloatTy::F32 => self.check_known_consts(cx, e, s, "f32"),
                FloatTy::F64 => self.check_known_consts(cx, e, s, "f64"),
            },
            LitKind::Float(s, LitFloatType::Unsuffixed) => self.check_known_consts(cx, e, s, "f{32, 64}"),
            _ => (),
        }
    }

    fn check_known_consts(&self, cx: &LateContext<'_>, e: &Expr<'_>, s: symbol::Symbol, module: &str) {
        let s = s.as_str();
        if s.parse::<f64>().is_ok() {
            for &(constant, name, min_digits, msrv) in &KNOWN_CONSTS {
                if is_approx_const(constant, s, min_digits)
                    && msrv.as_ref().map_or(true, |msrv| meets_msrv(self.msrv.as_ref(), msrv))
                {
                    span_lint_and_help(
                        cx,
                        APPROX_CONSTANT,
                        e.span,
                        &format!("approximate value of `{}::consts::{}` found", module, &name),
                        None,
                        "consider using the constant directly",
                    );
                    return;
                }
            }
        }
    }
}

impl_lint_pass!(ApproxConstant => [APPROX_CONSTANT]);

impl<'tcx> LateLintPass<'tcx> for ApproxConstant {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let ExprKind::Lit(lit) = &e.kind {
            self.check_lit(cx, &lit.node, e);
        }
    }

    extract_msrv_attr!(LateContext);
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
