use crate::{ImplTraitContext, ImplTraitPosition, LoweringContext};
use rustc_ast::{AttrVec, Block, BlockCheckMode, Expr, Local, LocalKind, Stmt, StmtKind};
use rustc_hir as hir;
use rustc_session::parse::feature_err;
use rustc_span::symbol::Ident;
use rustc_span::{sym, DesugaringKind};

use smallvec::SmallVec;

impl<'a, 'hir> LoweringContext<'a, 'hir> {
    pub(super) fn lower_block(
        &mut self,
        b: &Block,
        targeted_by_break: bool,
    ) -> &'hir hir::Block<'hir> {
        self.arena.alloc(self.lower_block_noalloc(b, targeted_by_break))
    }

    pub(super) fn lower_block_noalloc(
        &mut self,
        b: &Block,
        targeted_by_break: bool,
    ) -> hir::Block<'hir> {
        let (stmts, expr) = self.lower_stmts(&b.stmts);
        let rules = self.lower_block_check_mode(&b.rules);
        let hir_id = self.lower_node_id(b.id);
        hir::Block { hir_id, stmts, expr, rules, span: self.lower_span(b.span), targeted_by_break }
    }

    fn lower_stmts(
        &mut self,
        mut ast_stmts: &[Stmt],
    ) -> (&'hir [hir::Stmt<'hir>], Option<&'hir hir::Expr<'hir>>) {
        let mut stmts = SmallVec::<[hir::Stmt<'hir>; 8]>::new();
        let mut expr = None;
        while let [s, tail @ ..] = ast_stmts {
            match s.kind {
                StmtKind::Local(ref local) => {
                    let hir_id = self.lower_node_id(s.id);
                    match &local.kind {
                        LocalKind::InitElse(init, els) => {
                            let (s, e) = self.lower_let_else(hir_id, local, init, els, tail);
                            stmts.push(s);
                            expr = Some(e);
                            // remaining statements are in let-else expression
                            break;
                        }
                        _ => {
                            let local = self.lower_local(local);
                            self.alias_attrs(hir_id, local.hir_id);
                            let kind = hir::StmtKind::Local(local);
                            let span = self.lower_span(s.span);
                            stmts.push(hir::Stmt { hir_id, kind, span });
                        }
                    }
                }
                StmtKind::Item(ref it) => {
                    stmts.extend(self.lower_item_ref(it).into_iter().enumerate().map(
                        |(i, item_id)| {
                            let hir_id = match i {
                                0 => self.lower_node_id(s.id),
                                _ => self.next_id(),
                            };
                            let kind = hir::StmtKind::Item(item_id);
                            let span = self.lower_span(s.span);
                            hir::Stmt { hir_id, kind, span }
                        },
                    ));
                }
                StmtKind::Expr(ref e) => {
                    let e = self.lower_expr(e);
                    if tail.is_empty() {
                        expr = Some(e);
                    } else {
                        let hir_id = self.lower_node_id(s.id);
                        self.alias_attrs(hir_id, e.hir_id);
                        let kind = hir::StmtKind::Expr(e);
                        let span = self.lower_span(s.span);
                        stmts.push(hir::Stmt { hir_id, kind, span });
                    }
                }
                StmtKind::Semi(ref e) => {
                    let e = self.lower_expr(e);
                    let hir_id = self.lower_node_id(s.id);
                    self.alias_attrs(hir_id, e.hir_id);
                    let kind = hir::StmtKind::Semi(e);
                    let span = self.lower_span(s.span);
                    stmts.push(hir::Stmt { hir_id, kind, span });
                }
                StmtKind::Empty => {}
                StmtKind::MacCall(..) => panic!("shouldn't exist here"),
            }
            ast_stmts = &ast_stmts[1..];
        }
        (self.arena.alloc_from_iter(stmts), expr)
    }

    fn lower_local(&mut self, l: &Local) -> &'hir hir::Local<'hir> {
        let ty = l
            .ty
            .as_ref()
            .map(|t| self.lower_ty(t, ImplTraitContext::Disallowed(ImplTraitPosition::Binding)));
        let init = l.kind.init().map(|init| self.lower_expr(init));
        let hir_id = self.lower_node_id(l.id);
        let pat = self.lower_pat(&l.pat);
        let span = self.lower_span(l.span);
        let source = hir::LocalSource::Normal;
        self.lower_attrs(hir_id, &l.attrs);
        self.arena.alloc(hir::Local { hir_id, ty, pat, init, span, source })
    }

    fn lower_block_check_mode(&mut self, b: &BlockCheckMode) -> hir::BlockCheckMode {
        match *b {
            BlockCheckMode::Default => hir::BlockCheckMode::DefaultBlock,
            BlockCheckMode::Unsafe(u) => {
                hir::BlockCheckMode::UnsafeBlock(self.lower_unsafe_source(u))
            }
        }
    }

    fn lower_let_else(
        &mut self,
        stmt_hir_id: hir::HirId,
        local: &Local,
        init: &Expr,
        els: &Block,
        tail: &[Stmt],
    ) -> (hir::Stmt<'hir>, &'hir hir::Expr<'hir>) {
        let ty = local
            .ty
            .as_ref()
            .map(|t| self.lower_ty(t, ImplTraitContext::Disallowed(ImplTraitPosition::Binding)));
        let span = self.lower_span(local.span);
        let span = self.mark_span_with_reason(DesugaringKind::LetElse, span, None);
        let init = Some(self.lower_expr(init));
        let val = Ident::with_dummy_span(sym::val);
        let (pat, val_id) =
            self.pat_ident_binding_mode(span, val, hir::BindingAnnotation::Unannotated);
        let local_hir_id = self.lower_node_id(local.id);
        self.lower_attrs(local_hir_id, &local.attrs);
        // first statement which basically exists for the type annotation
        let stmt = {
            let local = self.arena.alloc(hir::Local {
                hir_id: local_hir_id,
                ty,
                pat,
                init,
                span,
                source: hir::LocalSource::Normal,
            });
            let kind = hir::StmtKind::Local(local);
            hir::Stmt { hir_id: stmt_hir_id, kind, span }
        };
        let let_expr = {
            let scrutinee = self.expr_ident(span, val, val_id);
            let let_kind = hir::ExprKind::Let(self.lower_pat(&local.pat), scrutinee, span);
            self.arena.alloc(self.expr(span, let_kind, AttrVec::new()))
        };
        let then_expr = {
            let (stmts, expr) = self.lower_stmts(tail);
            let block = self.block_all(span, stmts, expr);
            self.arena.alloc(self.expr_block(block, AttrVec::new()))
        };
        let else_expr = {
            let block = self.lower_block(els, false);
            self.arena.alloc(self.expr_block(block, AttrVec::new()))
        };
        self.alias_attrs(else_expr.hir_id, local_hir_id);
        let if_expr = self.arena.alloc(hir::Expr {
            hir_id: self.next_id(),
            span,
            kind: hir::ExprKind::If(let_expr, then_expr, Some(else_expr)),
        });
        if !self.sess.features_untracked().let_else {
            feature_err(
                &self.sess.parse_sess,
                sym::let_else,
                local.span,
                "`let...else` statements are unstable",
            )
            .emit();
        }
        (stmt, if_expr)
    }
}
