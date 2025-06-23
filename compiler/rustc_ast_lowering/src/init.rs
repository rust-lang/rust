use rustc_ast::ptr::P as AstP;
use rustc_ast::*;
use rustc_hir as hir;

use super::LoweringContext;

impl<'hir> LoweringContext<'_, 'hir> {
    pub(crate) fn lower_free_expr_init_tail(&mut self, expr: &Expr) -> hir::ExprKind<'hir> {
        if let ExprKind::Block(block, label) = &expr.kind {
            self.lower_block_init_tail(block, *label)
        } else {
            let expr = self.lower_expr(expr);
            hir::ExprKind::InitTail(self.arena.alloc(hir::InitKind::Free(expr)))
        }
    }

    /// This lower machine transforms a block expression within a in-place initialisation context
    pub(crate) fn lower_block_init_tail(
        &mut self,
        block: &Block,
        label: Option<Label>,
    ) -> hir::ExprKind<'hir> {
        let block = self.enter_init_tail_lowering(|this| this.lower_block(block, false));
        hir::ExprKind::InitTail(self.arena.alloc(hir::InitKind::Block(block, label)))
    }

    pub(crate) fn lower_implicit_init_tail(&mut self, expr: &Expr) -> hir::ExprKind<'hir> {
        match &expr.kind {
            ExprKind::Block(block, label) => self.lower_block_init_tail(block, *label),
            ExprKind::Array(exprs) => self.lower_array_init_tail(&exprs[..]),
            ExprKind::Struct(_) | ExprKind::Repeat(_, _) | ExprKind::Call(_, _) => {
                todo!(
                    "pinit: these are cases we still need to implement,
                    especially Call *must* be lowered to a `struct` init tail expr,
                    and block-style `struct` supports non-member fields so that computation
                    and initialisation can interleave"
                )
            }
            _ => self.lower_free_expr_init_tail(expr),
        }
    }

    pub(crate) fn lower_array_init_tail(&mut self, exprs: &[AstP<Expr>]) -> hir::ExprKind<'hir> {
        let exprs = self.arena.alloc_from_iter(exprs.iter().map(|expr| {
            let hir_id = self.next_id();
            let span = expr.span;
            self.lower_attrs(hir_id, &expr.attrs, span);
            let kind = self.lower_implicit_init_tail(expr);
            hir::Expr { hir_id, kind, span }
        }));
        hir::ExprKind::InitTail(self.arena.alloc(hir::InitKind::Array(exprs)))
    }

    pub(crate) fn lower_expr_init_tail(&mut self, kind: &InitKind) -> hir::ExprKind<'hir> {
        match kind {
            InitKind::Free(expr) => self.lower_free_expr_init_tail(expr),
            InitKind::Array(exprs) => self.lower_array_init_tail(exprs),
        }
    }
}
