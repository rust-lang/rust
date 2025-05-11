use clippy_config::Conf;
use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::SpanRangeExt;
use clippy_utils::{is_from_proc_macro, path_to_local};
use rustc_errors::Applicability;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::{BinOpKind, Constness, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, Lint, LintContext};
use rustc_middle::ty::TyCtxt;
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual `is_infinite` reimplementations
    /// (i.e., `x == <float>::INFINITY || x == <float>::NEG_INFINITY`).
    ///
    /// ### Why is this bad?
    /// The method `is_infinite` is shorter and more readable.
    ///
    /// ### Example
    /// ```no_run
    /// # let x = 1.0f32;
    /// if x == f32::INFINITY || x == f32::NEG_INFINITY {}
    /// ```
    /// Use instead:
    /// ```no_run
    /// # let x = 1.0f32;
    /// if x.is_infinite() {}
    /// ```
    #[clippy::version = "1.73.0"]
    pub MANUAL_IS_INFINITE,
    style,
    "use dedicated method to check if a float is infinite"
}
declare_clippy_lint! {
    /// ### What it does
    /// Checks for manual `is_finite` reimplementations
    /// (i.e., `x != <float>::INFINITY && x != <float>::NEG_INFINITY`).
    ///
    /// ### Why is this bad?
    /// The method `is_finite` is shorter and more readable.
    ///
    /// ### Example
    /// ```no_run
    /// # let x = 1.0f32;
    /// if x != f32::INFINITY && x != f32::NEG_INFINITY {}
    /// if x.abs() < f32::INFINITY {}
    /// ```
    /// Use instead:
    /// ```no_run
    /// # let x = 1.0f32;
    /// if x.is_finite() {}
    /// if x.is_finite() {}
    /// ```
    #[clippy::version = "1.73.0"]
    pub MANUAL_IS_FINITE,
    style,
    "use dedicated method to check if a float is finite"
}
impl_lint_pass!(ManualFloatMethods => [MANUAL_IS_INFINITE, MANUAL_IS_FINITE]);

#[derive(Clone, Copy)]
enum Variant {
    ManualIsInfinite,
    ManualIsFinite,
}

impl Variant {
    pub fn lint(self) -> &'static Lint {
        match self {
            Self::ManualIsInfinite => MANUAL_IS_INFINITE,
            Self::ManualIsFinite => MANUAL_IS_FINITE,
        }
    }

    pub fn msg(self) -> &'static str {
        match self {
            Self::ManualIsInfinite => "manually checking if a float is infinite",
            Self::ManualIsFinite => "manually checking if a float is finite",
        }
    }
}

pub struct ManualFloatMethods {
    msrv: Msrv,
}

impl ManualFloatMethods {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

fn is_not_const(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    match tcx.def_kind(def_id) {
        DefKind::Mod
        | DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Variant
        | DefKind::Trait
        | DefKind::TyAlias
        | DefKind::ForeignTy
        | DefKind::TraitAlias
        | DefKind::AssocTy
        | DefKind::Macro(..)
        | DefKind::Field
        | DefKind::LifetimeParam
        | DefKind::ExternCrate
        | DefKind::Use
        | DefKind::ForeignMod
        | DefKind::GlobalAsm
        | DefKind::Impl { .. }
        | DefKind::OpaqueTy
        | DefKind::SyntheticCoroutineBody
        | DefKind::TyParam => true,

        DefKind::AnonConst
        | DefKind::InlineConst
        | DefKind::Const
        | DefKind::ConstParam
        | DefKind::Static { .. }
        | DefKind::Ctor(..)
        | DefKind::AssocConst => false,

        DefKind::Fn | DefKind::AssocFn | DefKind::Closure => tcx.constness(def_id) == Constness::NotConst,
    }
}

impl<'tcx> LateLintPass<'tcx> for ManualFloatMethods {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let ExprKind::Binary(kind, lhs, rhs) = expr.kind
            && let ExprKind::Binary(lhs_kind, lhs_lhs, lhs_rhs) = lhs.kind
            && let ExprKind::Binary(rhs_kind, rhs_lhs, rhs_rhs) = rhs.kind
            // Checking all possible scenarios using a function would be a hopeless task, as we have
            // 16 possible alignments of constants/operands. For now, let's use `partition`.
            && let mut exprs = [lhs_lhs, lhs_rhs, rhs_lhs, rhs_rhs]
            && exprs.iter_mut().partition_in_place(|i| path_to_local(i).is_some()) == 2
            && !expr.span.in_external_macro(cx.sess().source_map())
            && (
                is_not_const(cx.tcx, cx.tcx.hir_enclosing_body_owner(expr.hir_id).into())
                    || self.msrv.meets(cx, msrvs::CONST_FLOAT_CLASSIFY)
            )
            && let [first, second, const_1, const_2] = exprs
            && let ecx = ConstEvalCtxt::new(cx)
            && let Some(const_1) = ecx.eval(const_1)
            && let Some(const_2) = ecx.eval(const_2)
            && path_to_local(first).is_some_and(|f| path_to_local(second).is_some_and(|s| f == s))
            // The actual infinity check, we also allow `NEG_INFINITY` before` INFINITY` just in
            // case somebody does that for some reason
            && (is_infinity(&const_1) && is_neg_infinity(&const_2)
                || is_neg_infinity(&const_1) && is_infinity(&const_2))
            && let Some(local_snippet) = first.span.get_source_text(cx)
        {
            let variant = match (kind.node, lhs_kind.node, rhs_kind.node) {
                (BinOpKind::Or, BinOpKind::Eq, BinOpKind::Eq) => Variant::ManualIsInfinite,
                (BinOpKind::And, BinOpKind::Ne, BinOpKind::Ne) => Variant::ManualIsFinite,
                _ => return,
            };
            if is_from_proc_macro(cx, expr) {
                return;
            }

            span_lint_and_then(cx, variant.lint(), expr.span, variant.msg(), |diag| {
                match variant {
                    Variant::ManualIsInfinite => {
                        diag.span_suggestion(
                            expr.span,
                            "use the dedicated method instead",
                            format!("{local_snippet}.is_infinite()"),
                            Applicability::MachineApplicable,
                        );
                    },
                    Variant::ManualIsFinite => {
                        // TODO: There's probably some better way to do this, i.e., create
                        // multiple suggestions with notes between each of them
                        diag.span_suggestion_verbose(
                            expr.span,
                            "use the dedicated method instead",
                            format!("{local_snippet}.is_finite()"),
                            Applicability::MaybeIncorrect,
                        )
                        .span_suggestion_verbose(
                            expr.span,
                            "this will alter how it handles NaN; if that is a problem, use instead",
                            format!("{local_snippet}.is_finite() || {local_snippet}.is_nan()"),
                            Applicability::MaybeIncorrect,
                        )
                        .span_suggestion_verbose(
                            expr.span,
                            "or, for conciseness",
                            format!("!{local_snippet}.is_infinite()"),
                            Applicability::MaybeIncorrect,
                        );
                    },
                }
            });
        }
    }
}

fn is_infinity(constant: &Constant<'_>) -> bool {
    match constant {
        // FIXME(f16_f128): add f16 and f128 when constants are available
        Constant::F32(float) => *float == f32::INFINITY,
        Constant::F64(float) => *float == f64::INFINITY,
        _ => false,
    }
}

fn is_neg_infinity(constant: &Constant<'_>) -> bool {
    match constant {
        // FIXME(f16_f128): add f16 and f128 when constants are available
        Constant::F32(float) => *float == f32::NEG_INFINITY,
        Constant::F64(float) => *float == f64::NEG_INFINITY,
        _ => false,
    }
}
