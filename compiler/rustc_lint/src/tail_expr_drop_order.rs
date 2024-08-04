use std::mem::swap;

use rustc_ast::UnOp;
use rustc_hir::def::Res;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{Block, Expr, ExprKind, LetStmt, Pat, PatKind, QPath, StmtKind};
use rustc_macros::LintDiagnostic;
use rustc_session::lint::FutureIncompatibilityReason;
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::edition::Edition;
use rustc_span::Span;

use crate::{LateContext, LateLintPass};

declare_lint! {
    /// The `tail_expr_drop_order` lint looks for those values generated at the tail expression location, that of type
    /// with a significant `Drop` implementation, such as locks.
    /// In case there are also local variables of type with significant `Drop` implementation as well,
    /// this lint warns you of a potential transposition in the drop order.
    /// Your discretion on the new drop order introduced by Edition 2024 is required.
    ///
    /// ### Example
    /// ```rust,edition2024
    /// #![feature(shorter_tail_lifetimes)]
    /// #![warn(tail_expr_drop_order)]
    /// struct Droppy;
    /// impl Droppy {
    ///     fn get(&self) -> i32 {
    ///         0
    ///     }
    /// }
    /// impl Drop for Droppy {
    ///     fn drop(&mut self) {
    ///         println!("loud drop");
    ///     }
    /// }
    /// fn edition_2024() -> i32 {
    ///     let another_droppy = Droppy;
    ///     Droppy.get()
    /// }
    /// fn main() {
    ///     edition_2024();
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// In tail expression of blocks or function bodies,
    /// values of type with significant `Drop` implementation has an ill-specified drop order before Edition 2024
    /// so that they are dropped only after dropping local variables.
    /// Edition 2024 introduces a new rule with drop orders for them, so that they are dropped first before dropping local variables.
    ///
    /// This lint only points out the issue with `Droppy`, which will be dropped before `another_droppy` does in Edition 2024.
    /// No fix will be proposed.
    /// However, the most probable fix is to hoist `Droppy` into its own local variable binding.
    /// ```rust
    /// struct Droppy;
    /// impl Droppy {
    ///     fn get(&self) -> i32 {
    ///         0
    ///     }
    /// }
    /// fn edition_2024() -> i32 {
    ///     let value = Droppy;
    ///     let another_droppy = Droppy;
    ///     value.get()
    /// }
    /// ```
    pub TAIL_EXPR_DROP_ORDER,
    Allow,
    "Detect and warn on significant change in drop order in tail expression location",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionSemanticsChange(Edition::Edition2024),
        reference: "issue #123739 <https://github.com/rust-lang/rust/issues/123739>",
    };
}

declare_lint_pass!(TailExprDropOrder => [TAIL_EXPR_DROP_ORDER]);

impl<'tcx> LateLintPass<'tcx> for TailExprDropOrder {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx Block<'tcx>) {
        LintVisitor { cx, locals: <_>::default() }.check_block_inner(block);
    }
}

struct LintVisitor<'tcx, 'a> {
    cx: &'a LateContext<'tcx>,
    locals: Vec<Span>,
}

struct LocalCollector<'tcx, 'a> {
    cx: &'a LateContext<'tcx>,
    locals: &'a mut Vec<Span>,
}

impl<'tcx, 'a> Visitor<'tcx> for LocalCollector<'tcx, 'a> {
    type Result = ();
    fn visit_pat(&mut self, pat: &'tcx Pat<'tcx>) {
        if let PatKind::Binding(_binding_mode, id, ident, pat) = pat.kind {
            let ty = self.cx.typeck_results().node_type(id);
            if ty.has_significant_drop(self.cx.tcx, self.cx.param_env) {
                self.locals.push(ident.span);
            }
            if let Some(pat) = pat {
                self.visit_pat(pat);
            }
        } else {
            intravisit::walk_pat(self, pat);
        }
    }
}

impl<'tcx, 'a> Visitor<'tcx> for LintVisitor<'tcx, 'a> {
    fn visit_block(&mut self, block: &'tcx Block<'tcx>) {
        let mut locals = <_>::default();
        swap(&mut locals, &mut self.locals);
        self.check_block_inner(block);
        swap(&mut locals, &mut self.locals);
    }
    fn visit_local(&mut self, local: &'tcx LetStmt<'tcx>) {
        LocalCollector { cx: self.cx, locals: &mut self.locals }.visit_local(local);
    }
}

impl<'tcx, 'a> LintVisitor<'tcx, 'a> {
    fn check_block_inner(&mut self, block: &Block<'tcx>) {
        if !block.span.at_least_rust_2024() {
            // We only lint for Edition 2024 onwards
            return;
        }
        let Some(tail_expr) = block.expr else { return };
        for stmt in block.stmts {
            match stmt.kind {
                StmtKind::Let(let_stmt) => self.visit_local(let_stmt),
                StmtKind::Item(_) => {}
                StmtKind::Expr(e) | StmtKind::Semi(e) => self.visit_expr(e),
            }
        }
        if self.locals.is_empty() {
            return;
        }
        LintTailExpr { cx: self.cx, locals: &self.locals }.visit_expr(tail_expr);
    }
}

struct LintTailExpr<'tcx, 'a> {
    cx: &'a LateContext<'tcx>,
    locals: &'a [Span],
}

impl<'tcx, 'a> LintTailExpr<'tcx, 'a> {
    fn expr_eventually_point_into_local(mut expr: &Expr<'tcx>) -> bool {
        loop {
            match expr.kind {
                ExprKind::Index(access, _, _) | ExprKind::Field(access, _) => expr = access,
                ExprKind::AddrOf(_, _, referee) | ExprKind::Unary(UnOp::Deref, referee) => {
                    expr = referee
                }
                ExprKind::Path(_)
                    if let ExprKind::Path(QPath::Resolved(_, path)) = expr.kind
                        && let [local, ..] = path.segments
                        && let Res::Local(_) = local.res =>
                {
                    return true;
                }
                _ => return false,
            }
        }
    }

    fn expr_generates_nonlocal_droppy_value(&self, expr: &Expr<'tcx>) -> bool {
        if Self::expr_eventually_point_into_local(expr) {
            return false;
        }
        self.cx.typeck_results().expr_ty(expr).has_significant_drop(self.cx.tcx, self.cx.param_env)
    }
}

impl<'tcx, 'a> Visitor<'tcx> for LintTailExpr<'tcx, 'a> {
    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        if self.expr_generates_nonlocal_droppy_value(expr) {
            self.cx.tcx.emit_node_span_lint(
                TAIL_EXPR_DROP_ORDER,
                expr.hir_id,
                expr.span,
                TailExprDropOrderLint { spans: self.locals.to_vec() },
            );
            return;
        }
        match expr.kind {
            ExprKind::ConstBlock(_)
            | ExprKind::Array(_)
            | ExprKind::Break(_, _)
            | ExprKind::Continue(_)
            | ExprKind::Ret(_)
            | ExprKind::Become(_)
            | ExprKind::Yield(_, _)
            | ExprKind::InlineAsm(_)
            | ExprKind::If(_, _, _)
            | ExprKind::Loop(_, _, _, _)
            | ExprKind::Match(_, _, _)
            | ExprKind::Closure(_)
            | ExprKind::DropTemps(_)
            | ExprKind::OffsetOf(_, _)
            | ExprKind::Assign(_, _, _)
            | ExprKind::AssignOp(_, _, _)
            | ExprKind::Lit(_)
            | ExprKind::Err(_) => {}

            ExprKind::MethodCall(_, _, _, _)
            | ExprKind::Call(_, _)
            | ExprKind::Type(_, _)
            | ExprKind::Tup(_)
            | ExprKind::Binary(_, _, _)
            | ExprKind::Unary(_, _)
            | ExprKind::Path(_)
            | ExprKind::Let(_)
            | ExprKind::Cast(_, _)
            | ExprKind::Field(_, _)
            | ExprKind::Index(_, _, _)
            | ExprKind::AddrOf(_, _, _)
            | ExprKind::Struct(_, _, _)
            | ExprKind::Repeat(_, _) => intravisit::walk_expr(self, expr),

            ExprKind::Block(block, _) => {
                LintVisitor { cx: self.cx, locals: <_>::default() }.check_block_inner(block)
            }
        }
    }
    fn visit_block(&mut self, block: &'tcx Block<'tcx>) {
        LintVisitor { cx: self.cx, locals: <_>::default() }.check_block_inner(block);
    }
}

#[derive(LintDiagnostic)]
#[diag(lint_tail_expr_drop_order)]
struct TailExprDropOrderLint {
    #[label]
    pub spans: Vec<Span>,
}
