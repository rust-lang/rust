use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::get_parent_expr;
use clippy_utils::source::snippet_with_applicability;
use if_chain::if_chain;
use rustc_ast::ast::{LitIntType, LitKind};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Block, Expr, ExprKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{Int, IntTy, Ty, Uint, UintTy};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for implicit saturating addition.
    ///
    /// ### Why is this bad?
    /// The built-in function is more readable and may be faster.
    ///
    /// ### Example
    /// ```rust
    ///let mut u:u32 = 7000;
    ///
    /// if u != u32::MAX {
    ///     u += 1;
    /// }
    /// ```
    /// Use instead:
    /// ```rust
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
        if_chain! {
            if let ExprKind::If(cond, then, None) = expr.kind;
            if let ExprKind::DropTemps(expr1) = cond.kind;
            if let Some((c, op_node, l)) = get_const(cx, expr1);
            if let BinOpKind::Ne | BinOpKind::Lt = op_node;
            if let ExprKind::Block(block, None) = then.kind;
            if let Block {
                stmts:
                    [Stmt
                        { kind: StmtKind::Expr(ex) | StmtKind::Semi(ex), .. }],
                        expr: None, ..} |
                        Block { stmts: [], expr: Some(ex), ..} = block;
            if let ExprKind::AssignOp(op1, target, value) = ex.kind;
            let ty = cx.typeck_results().expr_ty(target);
            if Some(c) == get_int_max(ty);
            if clippy_utils::SpanlessEq::new(cx).eq_expr(l, target);
            if BinOpKind::Add == op1.node;
            if let ExprKind::Lit(ref lit) = value.kind;
            if let LitKind::Int(1, LitIntType::Unsuffixed) = lit.node;
            if block.expr.is_none();
            then {
                let mut app = Applicability::MachineApplicable;
                let code = snippet_with_applicability(cx, target.span, "_", &mut app);
                let sugg = if let Some(parent) = get_parent_expr(cx, expr) && let ExprKind::If(_cond, _then, Some(else_)) = parent.kind && else_.hir_id == expr.hir_id {format!("{{{code} = {code}.saturating_add(1); }}")} else {format!("{code} = {code}.saturating_add(1);")};
                span_lint_and_sugg(cx, IMPLICIT_SATURATING_ADD, expr.span, "manual saturating add detected", "use instead", sugg, app);
            }
        }
    }
}

fn get_int_max(ty: Ty<'_>) -> Option<u128> {
    match ty.peel_refs().kind() {
        Int(IntTy::I8) => i8::max_value().try_into().ok(),
        Int(IntTy::I16) => i16::max_value().try_into().ok(),
        Int(IntTy::I32) => i32::max_value().try_into().ok(),
        Int(IntTy::I64) => i64::max_value().try_into().ok(),
        Int(IntTy::I128) => i128::max_value().try_into().ok(),
        Int(IntTy::Isize) => isize::max_value().try_into().ok(),
        Uint(UintTy::U8) => u8::max_value().try_into().ok(),
        Uint(UintTy::U16) => u16::max_value().try_into().ok(),
        Uint(UintTy::U32) => u32::max_value().try_into().ok(),
        Uint(UintTy::U64) => u64::max_value().try_into().ok(),
        Uint(UintTy::U128) => Some(u128::max_value()),
        Uint(UintTy::Usize) => usize::max_value().try_into().ok(),
        _ => None,
    }
}

fn get_const<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>) -> Option<(u128, BinOpKind, &'tcx Expr<'tcx>)> {
    if let ExprKind::Binary(op, l, r) = expr.kind {
        let tr = cx.typeck_results();
        if let Some((Constant::Int(c), _)) = constant(cx, tr, r) {
            return Some((c, op.node, l));
        };
        if let Some((Constant::Int(c), _)) = constant(cx, tr, l) {
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
