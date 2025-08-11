use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::msrvs::{self, Msrv};
use rustc_ast::ast::{FloatTy, LitFloatType, LitKind};
use rustc_hir::{HirId, Lit, RustcVersion};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::{Span, symbol};
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
    /// ```no_run
    /// let x = 3.14;
    /// let y = 1_f64 / x;
    /// ```
    /// Use instead:
    /// ```no_run
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
    msrv: Msrv,
}

impl ApproxConstant {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl LateLintPass<'_> for ApproxConstant {
    fn check_lit(&mut self, cx: &LateContext<'_>, _hir_id: HirId, lit: Lit, _negated: bool) {
        match lit.node {
            LitKind::Float(s, LitFloatType::Suffixed(fty)) => match fty {
                FloatTy::F16 => self.check_known_consts(cx, lit.span, s, "f16"),
                FloatTy::F32 => self.check_known_consts(cx, lit.span, s, "f32"),
                FloatTy::F64 => self.check_known_consts(cx, lit.span, s, "f64"),
                FloatTy::F128 => self.check_known_consts(cx, lit.span, s, "f128"),
            },
            // FIXME(f16_f128): add `f16` and `f128` when these types become stable.
            LitKind::Float(s, LitFloatType::Unsuffixed) => self.check_known_consts(cx, lit.span, s, "f{32, 64}"),
            _ => (),
        }
    }
}

impl ApproxConstant {
    fn check_known_consts(&self, cx: &LateContext<'_>, span: Span, s: symbol::Symbol, module: &str) {
        let s = s.as_str();
        if let Ok(maybe_constant) = s.parse::<f64>() {
            for &(constant, name, min_digits, msrv) in &KNOWN_CONSTS {
                if is_approx_const(constant, s, maybe_constant, min_digits)
                    && msrv.is_none_or(|msrv| self.msrv.meets(cx, msrv))
                {
                    span_lint_and_help(
                        cx,
                        APPROX_CONSTANT,
                        span,
                        format!("approximate value of `{module}::consts::{name}` found"),
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

fn count_digits_after_dot(input: &str) -> usize {
    input
        .char_indices()
        .find(|(_, ch)| *ch == '.')
        .map_or(0, |(i, _)| input.len() - i - 1)
}

/// Returns `false` if the number of significant figures in `value` are
/// less than `min_digits`; otherwise, returns true if `value` is equal
/// to `constant`, rounded to the number of significant digits present in `value`.
#[must_use]
fn is_approx_const(constant: f64, value: &str, f_value: f64, min_digits: usize) -> bool {
    if value.len() <= min_digits {
        // The value is not precise enough
        false
    } else if f_value.to_string().len() > min_digits && constant.to_string().starts_with(&f_value.to_string()) {
        // The value represents the same value
        true
    } else {
        // The value is a truncated constant

        // Print constant with numeric formatting (`0`), with the length of `value` as minimum width
        // (`value_len$`), and with the same precision as `value` (`.value_prec$`).
        // See https://doc.rust-lang.org/std/fmt/index.html.
        let round_const = format!(
            "{constant:0value_len$.value_prec$}",
            value_len = value.len(),
            value_prec = count_digits_after_dot(value)
        );
        value == round_const
    }
}
