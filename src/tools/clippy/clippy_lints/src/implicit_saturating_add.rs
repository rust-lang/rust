use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::get_parent_expr;
use clippy_utils::source::snippet_with_context;
use rustc_ast::ast::{LitIntType, LitKind};
use rustc_data_structures::packed::Pu128;
use rustc_errors::Applicability;
use rustc_hir::{AssignOpKind, BinOpKind, Block, Expr, ExprKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{IntTy, Ty, UintTy};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for implicit saturating addition.
    ///
    /// ### Why is this bad?
    /// The built-in function is more readable and may be faster.
    ///
    /// ### Example
    /// ```no_run
    ///let mut u:u32 = 7000;
    ///
    /// if u != u32::MAX {
    ///     u += 1;
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    ///let mut u:u32 = 7000;
    ///
    /// u = u.saturating_add(1);
    /// ```
    #[clippy::version = "1.66.0"]
    pub IMPLICIT_SATURATING_ADD,
    style,
    "Perform saturating addition instead of implicitly checking max bound of data type"
}
declare_lint_pass!(ImplicitSaturatingAdd => [IMPLICIT_SATURATING_ADD]);

impl<'tcx> LateLintPass<'tcx> for ImplicitSaturatingAdd {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let ExprKind::If(cond, then, None) = expr.kind
            && let Some((c, op_node, l)) = get_const(cx, cond)
            && let BinOpKind::Ne | BinOpKind::Lt = op_node
            && let ExprKind::Block(block, None) = then.kind
            && let Block {
                stmts:
                    [
                        Stmt {
                            kind: StmtKind::Expr(ex) | StmtKind::Semi(ex),
                            ..
                        },
                    ],
                expr: None,
                ..
            }
            | Block {
                stmts: [],
                expr: Some(ex),
                ..
            } = block
            && let ExprKind::AssignOp(op1, target, value) = ex.kind
            && let ty = cx.typeck_results().expr_ty(target)
            && Some(c) == get_int_max(ty)
            && let ctxt = expr.span.ctxt()
            && ex.span.ctxt() == ctxt
            && cond.span.ctxt() == ctxt
            && clippy_utils::SpanlessEq::new(cx).eq_expr(l, target)
            && AssignOpKind::AddAssign == op1.node
            && let ExprKind::Lit(lit) = value.kind
            && let LitKind::Int(Pu128(1), LitIntType::Unsuffixed) = lit.node
            && block.expr.is_none()
        {
            let mut app = Applicability::MachineApplicable;
            let code = snippet_with_context(cx, target.span, ctxt, "_", &mut app).0;
            let sugg = if let Some(parent) = get_parent_expr(cx, expr)
                && let ExprKind::If(_cond, _then, Some(else_)) = parent.kind
                && else_.hir_id == expr.hir_id
            {
                format!("{{{code} = {code}.saturating_add(1); }}")
            } else {
                format!("{code} = {code}.saturating_add(1);")
            };
            span_lint_and_sugg(
                cx,
                IMPLICIT_SATURATING_ADD,
                expr.span,
                "manual saturating add detected",
                "use instead",
                sugg,
                app,
            );
        }
    }
}

fn get_int_max(ty: Ty<'_>) -> Option<u128> {
    use rustc_middle::ty::{Int, Uint};
    match ty.peel_refs().kind() {
        Int(IntTy::I8) => i8::MAX.try_into().ok(),
        Int(IntTy::I16) => i16::MAX.try_into().ok(),
        Int(IntTy::I32) => i32::MAX.try_into().ok(),
        Int(IntTy::I64) => i64::MAX.try_into().ok(),
        Int(IntTy::I128) => i128::MAX.try_into().ok(),
        Int(IntTy::Isize) => isize::MAX.try_into().ok(),
        Uint(UintTy::U8) => Some(u8::MAX.into()),
        Uint(UintTy::U16) => Some(u16::MAX.into()),
        Uint(UintTy::U32) => Some(u32::MAX.into()),
        Uint(UintTy::U64) => Some(u64::MAX.into()),
        Uint(UintTy::U128) => Some(u128::MAX),
        Uint(UintTy::Usize) => usize::MAX.try_into().ok(),
        _ => None,
    }
}

fn get_const<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>) -> Option<(u128, BinOpKind, &'tcx Expr<'tcx>)> {
    if let ExprKind::Binary(op, l, r) = expr.kind {
        let ecx = ConstEvalCtxt::new(cx);
        if let Some(Constant::Int(c)) = ecx.eval(r) {
            return Some((c, op.node, l));
        }
        if let Some(Constant::Int(c)) = ecx.eval(l) {
            return Some((c, invert_op(op.node)?, r));
        }
    }
    None
}

fn invert_op(op: BinOpKind) -> Option<BinOpKind> {
    use rustc_hir::BinOpKind::{Ge, Gt, Le, Lt, Ne};
    match op {
        Lt => Some(Gt),
        Le => Some(Ge),
        Ne => Some(Ne),
        Ge => Some(Le),
        Gt => Some(Lt),
        _ => None,
    }
}
