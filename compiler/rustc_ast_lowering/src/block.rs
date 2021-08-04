use crate::{ImplTraitContext, ImplTraitPosition, LoweringContext};
use rustc_ast::{Block, BlockCheckMode, Local, Stmt, StmtKind};
use rustc_hir as hir;

use smallvec::{smallvec, SmallVec};

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
        let (stmts, expr) = match &*b.stmts {
            [stmts @ .., Stmt { kind: StmtKind::Expr(e), .. }] => (stmts, Some(&*e)),
            stmts => (stmts, None),
        };
        let stmts = self.arena.alloc_from_iter(stmts.iter().flat_map(|stmt| self.lower_stmt(stmt)));
        let expr = expr.map(|e| self.lower_expr(e));
        let rules = self.lower_block_check_mode(&b.rules);
        let hir_id = self.lower_node_id(b.id);

        hir::Block { hir_id, stmts, expr, rules, span: self.lower_span(b.span), targeted_by_break }
    }

    fn lower_stmt(&mut self, s: &Stmt) -> SmallVec<[hir::Stmt<'hir>; 1]> {
        let (hir_id, kind) = match s.kind {
            StmtKind::Local(ref l) => {
                let l = self.lower_local(l);
                let hir_id = self.lower_node_id(s.id);
                self.alias_attrs(hir_id, l.hir_id);
                return smallvec![hir::Stmt {
                    hir_id,
                    kind: hir::StmtKind::Local(self.arena.alloc(l)),
                    span: self.lower_span(s.span),
                }];
            }
            StmtKind::Item(ref it) => {
                // Can only use the ID once.
                let mut id = Some(s.id);
                return self
                    .lower_item_id(it)
                    .into_iter()
                    .map(|item_id| {
                        let hir_id = id
                            .take()
                            .map(|id| self.lower_node_id(id))
                            .unwrap_or_else(|| self.next_id());

                        hir::Stmt {
                            hir_id,
                            kind: hir::StmtKind::Item(item_id),
                            span: self.lower_span(s.span),
                        }
                    })
                    .collect();
            }
            StmtKind::Expr(ref e) => {
                let e = self.lower_expr(e);
                let hir_id = self.lower_node_id(s.id);
                self.alias_attrs(hir_id, e.hir_id);
                (hir_id, hir::StmtKind::Expr(e))
            }
            StmtKind::Semi(ref e) => {
                let e = self.lower_expr(e);
                let hir_id = self.lower_node_id(s.id);
                self.alias_attrs(hir_id, e.hir_id);
                (hir_id, hir::StmtKind::Semi(e))
            }
            StmtKind::Empty => return smallvec![],
            StmtKind::MacCall(..) => panic!("shouldn't exist here"),
        };
        smallvec![hir::Stmt { hir_id, kind, span: self.lower_span(s.span) }]
    }

    fn lower_local(&mut self, l: &Local) -> hir::Local<'hir> {
        let ty = l
            .ty
            .as_ref()
            .map(|t| self.lower_ty(t, ImplTraitContext::Disallowed(ImplTraitPosition::Binding)));
        let init = l.kind.init().map(|init| self.lower_expr(init));
        let hir_id = self.lower_node_id(l.id);
        self.lower_attrs(hir_id, &l.attrs);
        hir::Local {
            hir_id,
            ty,
            pat: self.lower_pat(&l.pat),
            init,
            span: self.lower_span(l.span),
            source: hir::LocalSource::Normal,
        }
    }

    fn lower_block_check_mode(&mut self, b: &BlockCheckMode) -> hir::BlockCheckMode {
        match *b {
            BlockCheckMode::Default => hir::BlockCheckMode::DefaultBlock,
            BlockCheckMode::Unsafe(u) => {
                hir::BlockCheckMode::UnsafeBlock(self.lower_unsafe_source(u))
            }
        }
    }
}
