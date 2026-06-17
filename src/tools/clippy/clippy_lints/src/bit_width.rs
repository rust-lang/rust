use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_context;
use clippy_utils::{is_from_proc_macro, sym};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::{self, Ty};
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `T::BITS - x.leading_zeros()` when `x.bit_width()` is available.
    ///
    /// ### Why is this bad?
    ///  Manual reimplementations of `bit_width` increase code complexity for little benefit.
    ///
    /// ### Example
    /// ```no_run
    /// let x: u32 = 5;
    /// let bit_width = u32::BITS - x.leading_zeros();
    /// ```
    /// Use instead:
    /// ```no_run
    /// let x: u32 = 5;
    /// let bit_width = x.bit_width();
    /// ```
    #[clippy::version = "1.98.0"]
    pub MANUAL_BIT_WIDTH,
    pedantic,
    "manually reimplementing `bit_width`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `T::BITS - x.leading_zeros()` where T and x are of different types.
    ///
    /// ### Why is this bad?
    /// Substracting `leading_zeros` from the number of bits of another type might be
    /// a buggy implementation of the `bit_width` method.
    ///
    /// ### Example
    /// ```no_run
    /// let x: u64 = 5;
    /// let bit_width = u32::BITS - x.leading_zeros();
    /// ```
    /// Use instead:
    /// ```no_run
    /// let x: u64 = 5;
    /// let bit_width = x.bit_width();
    /// ```
    #[clippy::version = "1.98.0"]
    pub MISMATCHED_BIT_WIDTH_TYPE,
    suspicious,
    "type mismatch in bit width calculation"
}

impl_lint_pass!(ManualBitWidth => [MANUAL_BIT_WIDTH, MISMATCHED_BIT_WIDTH_TYPE]);

#[derive(Clone, Copy, PartialEq)]
enum IntKind<'a> {
    Int(ty::IntTy),
    Uint(ty::UintTy),
    // NOTE: in the following two variants, the inner `Ty` stores the entire `NonZero<T>`
    // and not just `T`. This is so that we can print it in the suggestion.
    NonZero(Ty<'a>),
    NonZeroU(Ty<'a>),
}

impl IntKind<'_> {
    fn inner_ty(self) -> String {
        match self {
            Self::Int(ty) => ty.name_str().to_string(),
            Self::Uint(ty) => ty.name_str().to_string(),
            Self::NonZero(ty) | Self::NonZeroU(ty) => ty.to_string(),
        }
    }

    fn suggestion(&self) -> &'static str {
        match self {
            Self::Int(_) => ".cast_unsigned().bit_width()",
            Self::Uint(_) => ".bit_width()",
            Self::NonZero(_) => ".cast_unsigned().bit_width().get()",
            Self::NonZeroU(_) => ".bit_width().get()",
        }
    }
}

pub struct ManualBitWidth {
    msrv: Msrv,
}

impl ManualBitWidth {
    pub fn new(conf: &Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl LateLintPass<'_> for ManualBitWidth {
    fn check_expr<'tcx>(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        if expr.span.in_external_macro(cx.sess().source_map()) {
            return;
        }

        match expr.kind {
            // `T::BITS - n.leading_zeros()`
            ExprKind::Binary(op, left, right)
                if op.node == BinOpKind::Sub
                    && let ExprKind::MethodCall(leading_zeros, recv, [], _) = right.kind
                    && leading_zeros.ident.name == sym::leading_zeros
                    && let ExprKind::Path(QPath::TypeRelative(hir_ty, segment)) = left.kind
                    && segment.ident.name == sym::BITS
                    && let right_ty = cx.typeck_results.expr_ty(recv)
                    && let Some(right_int_kind) = get_int_kind(cx, right_ty)
                    && let left_ty = cx.typeck_results.node_type(hir_ty.hir_id)
                    && let Some(left_int_kind) = get_int_kind(cx, left_ty)
                    && self.msrv.meets(cx, msrvs::BIT_WIDTH)
                    && left.span.eq_ctxt(right.span)
                    && !is_from_proc_macro(cx, expr) =>
            {
                if left_int_kind == right_int_kind {
                    // manual implementation of bit_width
                    emit_manual_bit_width(cx, recv, expr, right_int_kind);
                } else {
                    // mismatched calling types
                    emit_type_mismatch(cx, recv, expr, right_int_kind);
                }
            },
            _ => {},
        }
    }
}

fn get_int_kind<'a>(cx: &LateContext<'a>, ty: Ty<'a>) -> Option<IntKind<'a>> {
    match ty.kind() {
        // int::BITS or uint::BITS
        ty::Int(int_ty) => Some(IntKind::Int(*int_ty)),
        ty::Uint(uint_ty) => Some(IntKind::Uint(*uint_ty)),
        // NonZero::<int/uint>::BITS
        ty::Adt(adt, args) if cx.tcx.is_diagnostic_item(sym::NonZero, adt.did()) => {
            let arg = args.type_at(0);
            match arg.kind() {
                ty::Int(_) => Some(IntKind::NonZero(ty)),
                ty::Uint(_) => Some(IntKind::NonZeroU(ty)),
                _ => None,
            }
        },
        _ => None,
    }
}

fn emit_manual_bit_width(cx: &LateContext<'_>, recv: &Expr<'_>, full_expr: &Expr<'_>, ty_kind: IntKind<'_>) {
    span_lint_and_then(
        cx,
        MANUAL_BIT_WIDTH,
        full_expr.span,
        "manual implementation of `bit_width`",
        |diag| {
            let mut app = Applicability::MachineApplicable;
            let (recv_snip, _) = snippet_with_context(cx, recv.span, full_expr.span.ctxt(), "_", &mut app);
            let suggestion = ty_kind.suggestion();

            diag.span_suggestion_verbose(full_expr.span, "try", format!("{recv_snip}{suggestion}"), app);
        },
    );
}

fn emit_type_mismatch(cx: &LateContext<'_>, recv: &Expr<'_>, full_expr: &Expr<'_>, ty_kind: IntKind<'_>) {
    span_lint_and_then(
        cx,
        MISMATCHED_BIT_WIDTH_TYPE,
        full_expr.span,
        "possible buggy implementation of `bit_width`",
        |diag| {
            diag.note("in order to calculate the bit width, `T::BITS` should match the type of the value calling `.leading_zeros()`");

            let mut app = Applicability::MaybeIncorrect;
            let (recv_snip, _) = snippet_with_context(cx, recv.span, full_expr.span.ctxt(), "_", &mut app);
            let suggestion = ty_kind.suggestion();
            let x_ty = ty_kind.inner_ty();

            diag.span_suggestion_verbose(
                full_expr.span,
                format!("if you meant to use `{x_ty}::BITS`, use"),
                format!("{recv_snip}{suggestion}"),
                app,
            );
        },
    );
}
