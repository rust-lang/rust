use super::ARITHMETIC_SIDE_EFFECTS;
use clippy_utils::{
    consts::{constant, constant_simple, Constant},
    diagnostics::span_lint,
    is_lint_allowed, peel_hir_expr_refs, peel_hir_expr_unary,
};
use rustc_ast as ast;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Ty;
use rustc_session::impl_lint_pass;
use rustc_span::source_map::{Span, Spanned};

const HARD_CODED_ALLOWED_BINARY: &[[&str; 2]] = &[
    ["f32", "f32"],
    ["f64", "f64"],
    ["std::num::Saturating", "std::num::Saturating"],
    ["std::num::Wrapping", "std::num::Wrapping"],
    ["std::string::String", "&str"],
];
const HARD_CODED_ALLOWED_UNARY: &[&str] = &["f32", "f64", "std::num::Saturating", "std::num::Wrapping"];

#[derive(Debug)]
pub struct ArithmeticSideEffects {
    allowed_binary: FxHashMap<String, FxHashSet<String>>,
    allowed_unary: FxHashSet<String>,
    // Used to check whether expressions are constants, such as in enum discriminants and consts
    const_span: Option<Span>,
    expr_span: Option<Span>,
}

impl_lint_pass!(ArithmeticSideEffects => [ARITHMETIC_SIDE_EFFECTS]);

impl ArithmeticSideEffects {
    #[must_use]
    pub fn new(user_allowed_binary: Vec<[String; 2]>, user_allowed_unary: Vec<String>) -> Self {
        let mut allowed_binary: FxHashMap<String, FxHashSet<String>> = <_>::default();
        for [lhs, rhs] in user_allowed_binary.into_iter().chain(
            HARD_CODED_ALLOWED_BINARY
                .iter()
                .copied()
                .map(|[lhs, rhs]| [lhs.to_string(), rhs.to_string()]),
        ) {
            allowed_binary.entry(lhs).or_default().insert(rhs);
        }
        let allowed_unary = user_allowed_unary
            .into_iter()
            .chain(HARD_CODED_ALLOWED_UNARY.iter().copied().map(String::from))
            .collect();
        Self {
            allowed_binary,
            allowed_unary,
            const_span: None,
            expr_span: None,
        }
    }

    /// Checks if the lhs and the rhs types of a binary operation like "addition" or
    /// "multiplication" are present in the inner set of allowed types.
    fn has_allowed_binary(&self, lhs_ty: Ty<'_>, rhs_ty: Ty<'_>) -> bool {
        let lhs_ty_string = lhs_ty.to_string();
        let lhs_ty_string_elem = lhs_ty_string.split('<').next().unwrap_or_default();
        let rhs_ty_string = rhs_ty.to_string();
        let rhs_ty_string_elem = rhs_ty_string.split('<').next().unwrap_or_default();
        if let Some(rhs_from_specific) = self.allowed_binary.get(lhs_ty_string_elem)
            && {
                let rhs_has_allowed_ty = rhs_from_specific.contains(rhs_ty_string_elem);
                rhs_has_allowed_ty || rhs_from_specific.contains("*")
            }
        {
           true
        } else if let Some(rhs_from_glob) = self.allowed_binary.get("*") {
            rhs_from_glob.contains(rhs_ty_string_elem)
        } else {
            false
        }
    }

    /// Checks if the type of an unary operation like "negation" is present in the inner set of
    /// allowed types.
    fn has_allowed_unary(&self, ty: Ty<'_>) -> bool {
        let ty_string = ty.to_string();
        let ty_string_elem = ty_string.split('<').next().unwrap_or_default();
        self.allowed_unary.contains(ty_string_elem)
    }

    // For example, 8i32 or &i64::MAX.
    fn is_integral(ty: Ty<'_>) -> bool {
        ty.peel_refs().is_integral()
    }

    // Common entry-point to avoid code duplication.
    fn issue_lint(&mut self, cx: &LateContext<'_>, expr: &hir::Expr<'_>) {
        let msg = "arithmetic operation that can potentially result in unexpected side-effects";
        span_lint(cx, ARITHMETIC_SIDE_EFFECTS, expr.span, msg);
        self.expr_span = Some(expr.span);
    }

    /// Returns the numeric value of a literal integer originated from `expr`, if any.
    ///
    /// Literal integers can be originated from adhoc declarations like `1`, associated constants
    /// like `i32::MAX` or constant references like `N` from `const N: i32 = 1;`,
    fn literal_integer(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> Option<u128> {
        let actual = peel_hir_expr_unary(expr).0;
        if let hir::ExprKind::Lit(ref lit) = actual.kind && let ast::LitKind::Int(n, _) = lit.node {
            return Some(n)
        }
        if let Some((Constant::Int(n), _)) = constant(cx, cx.typeck_results(), expr) {
            return Some(n);
        }
        None
    }

    /// Manages when the lint should be triggered. Operations in constant environments, hard coded
    /// types, custom allowed types and non-constant operations that won't overflow are ignored.
    fn manage_bin_ops<'tcx>(
        &mut self,
        cx: &LateContext<'tcx>,
        expr: &hir::Expr<'tcx>,
        op: &Spanned<hir::BinOpKind>,
        lhs: &hir::Expr<'tcx>,
        rhs: &hir::Expr<'tcx>,
    ) {
        if constant_simple(cx, cx.typeck_results(), expr).is_some() {
            return;
        }
        if !matches!(
            op.node,
            hir::BinOpKind::Add
                | hir::BinOpKind::Div
                | hir::BinOpKind::Mul
                | hir::BinOpKind::Rem
                | hir::BinOpKind::Shl
                | hir::BinOpKind::Shr
                | hir::BinOpKind::Sub
        ) {
            return;
        };
        let lhs_ty = cx.typeck_results().expr_ty(lhs);
        let rhs_ty = cx.typeck_results().expr_ty(rhs);
        if self.has_allowed_binary(lhs_ty, rhs_ty) {
            return;
        }
        let has_valid_op = if Self::is_integral(lhs_ty) && Self::is_integral(rhs_ty) {
            let (actual_lhs, lhs_ref_counter) = peel_hir_expr_refs(lhs);
            let (actual_rhs, rhs_ref_counter) = peel_hir_expr_refs(rhs);
            match (
                Self::literal_integer(cx, actual_lhs),
                Self::literal_integer(cx, actual_rhs),
            ) {
                (None, None) => false,
                (None, Some(n)) | (Some(n), None) => match (&op.node, n) {
                    (hir::BinOpKind::Div | hir::BinOpKind::Rem, 0) => false,
                    (hir::BinOpKind::Add | hir::BinOpKind::Sub, 0)
                    | (hir::BinOpKind::Div | hir::BinOpKind::Rem, _)
                    | (hir::BinOpKind::Mul, 0 | 1) => true,
                    _ => false,
                },
                (Some(_), Some(_)) => {
                    matches!((lhs_ref_counter, rhs_ref_counter), (0, 0))
                },
            }
        } else {
            false
        };
        if !has_valid_op {
            self.issue_lint(cx, expr);
        }
    }

    fn manage_unary_ops<'tcx>(
        &mut self,
        cx: &LateContext<'tcx>,
        expr: &hir::Expr<'tcx>,
        un_expr: &hir::Expr<'tcx>,
        un_op: hir::UnOp,
    ) {
        let hir::UnOp::Neg = un_op else { return; };
        if constant(cx, cx.typeck_results(), un_expr).is_some() {
            return;
        }
        let ty = cx.typeck_results().expr_ty(expr).peel_refs();
        if self.has_allowed_unary(ty) {
            return;
        }
        let actual_un_expr = peel_hir_expr_refs(un_expr).0;
        if Self::literal_integer(cx, actual_un_expr).is_some() {
            return;
        }
        self.issue_lint(cx, expr);
    }

    fn should_skip_expr(&mut self, cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
        is_lint_allowed(cx, ARITHMETIC_SIDE_EFFECTS, expr.hir_id)
            || self.expr_span.is_some()
            || self.const_span.map_or(false, |sp| sp.contains(expr.span))
    }
}

impl<'tcx> LateLintPass<'tcx> for ArithmeticSideEffects {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &hir::Expr<'tcx>) {
        if self.should_skip_expr(cx, expr) {
            return;
        }
        match &expr.kind {
            hir::ExprKind::AssignOp(op, lhs, rhs) | hir::ExprKind::Binary(op, lhs, rhs) => {
                self.manage_bin_ops(cx, expr, op, lhs, rhs);
            },
            hir::ExprKind::Unary(un_op, un_expr) => {
                self.manage_unary_ops(cx, expr, un_expr, *un_op);
            },
            _ => {},
        }
    }

    fn check_body(&mut self, cx: &LateContext<'_>, body: &hir::Body<'_>) {
        let body_owner = cx.tcx.hir().body_owner(body.id());
        let body_owner_def_id = cx.tcx.hir().body_owner_def_id(body.id());

        let body_owner_kind = cx.tcx.hir().body_owner_kind(body_owner_def_id);
        if let hir::BodyOwnerKind::Const | hir::BodyOwnerKind::Static(_) = body_owner_kind {
            let body_span = cx.tcx.hir().span_with_body(body_owner);
            if let Some(span) = self.const_span && span.contains(body_span) {
                return;
            }
            self.const_span = Some(body_span);
        }
    }

    fn check_body_post(&mut self, cx: &LateContext<'_>, body: &hir::Body<'_>) {
        let body_owner = cx.tcx.hir().body_owner(body.id());
        let body_span = cx.tcx.hir().span(body_owner);
        if let Some(span) = self.const_span && span.contains(body_span) {
            return;
        }
        self.const_span = None;
    }

    fn check_expr_post(&mut self, _: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if Some(expr.span) == self.expr_span {
            self.expr_span = None;
        }
    }
}
