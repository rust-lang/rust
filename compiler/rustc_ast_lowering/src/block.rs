use rustc_ast::{Block, BlockCheckMode, Local, LocalKind, Stmt, StmtKind};
use rustc_hir as hir;
use rustc_span::sym;
use smallvec::SmallVec;

use crate::{ImplTraitContext, ImplTraitPosition, LoweringContext};

impl<'a, 'hir> LoweringContext<'a, 'hir> {
    pub(super) fn lower_block(
        &mut self,
        b: &Block,
        targeted_by_break: bool,
    ) -> &'hir hir::Block<'hir> {
        let hir_id = self.lower_node_id(b.id);
        self.arena.alloc(self.lower_block_noalloc(hir_id, b, targeted_by_break))
    }

    pub(super) fn lower_block_noalloc(
        &mut self,
        hir_id: hir::HirId,
        b: &Block,
        targeted_by_break: bool,
    ) -> hir::Block<'hir> {
        let (stmts, expr) = self.lower_stmts(&b.stmts);
        let rules = self.lower_block_check_mode(&b.rules);
        hir::Block { hir_id, stmts, expr, rules, span: self.lower_span(b.span), targeted_by_break }
    }

    fn lower_one_stmt(
        &mut self,
        s: &Stmt,
        at_tail: Option<&mut Option<&'hir hir::Expr<'hir>>>,
        hir_stmts: &mut impl Extend<hir::Stmt<'hir>>,
    ) {
        match &s.kind {
            StmtKind::Let(local) => {
                let hir_id = self.lower_node_id(s.id);
                let local = self.lower_local(local);
                self.alias_attrs(hir_id, local.hir_id);
                let kind = hir::StmtKind::Let(local);
                let span = self.lower_span(s.span);
                hir_stmts.extend([hir::Stmt { hir_id, kind, span }]);
            }
            StmtKind::Item(it) => {
                hir_stmts.extend(self.lower_item_ref(it).into_iter().enumerate().map(
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
            StmtKind::Expr(e) => {
                if let Some(expr) = at_tail {
                    let e = if self.is_in_init_tail {
                        let hir_id = self.lower_node_id(s.id);
                        let kind = self.lower_implicit_init_tail(e);
                        self.arena.alloc(hir::Expr { hir_id, kind, span: s.span })
                    } else {
                        self.lower_expr(e)
                    };
                    *expr = Some(e);
                } else {
                    let e = self.lower_expr(e);
                    let hir_id = self.lower_node_id(s.id);
                    self.alias_attrs(hir_id, e.hir_id);
                    let kind = hir::StmtKind::Expr(e);
                    let span = self.lower_span(s.span);
                    hir_stmts.extend([hir::Stmt { hir_id, kind, span }]);
                }
            }
            StmtKind::Semi(e) => {
                let e = self.lower_expr(e);
                let hir_id = self.lower_node_id(s.id);
                self.alias_attrs(hir_id, e.hir_id);
                let kind = hir::StmtKind::Semi(e);
                let span = self.lower_span(s.span);
                hir_stmts.extend([hir::Stmt { hir_id, kind, span }]);
            }
            StmtKind::Empty => {}
            StmtKind::MacCall(..) => panic!("shouldn't exist here"),
        }
    }

    fn lower_stmts(
        &mut self,
        ast_stmts: &[Stmt],
    ) -> (&'hir [hir::Stmt<'hir>], Option<&'hir hir::Expr<'hir>>) {
        let mut hir_stmts = SmallVec::<[hir::Stmt<'hir>; 8]>::new();
        let mut expr = None;
        if let [stmts @ .., tail] = ast_stmts {
            self.enter_non_init_tail_lowering(|this| {
                for s in stmts {
                    this.lower_one_stmt(s, None, &mut hir_stmts);
                }
            });
            self.lower_one_stmt(tail, Some(&mut expr), &mut hir_stmts);
        }
        (self.arena.alloc_from_iter(hir_stmts), expr)
    }

    /// Return an `ImplTraitContext` that allows impl trait in bindings if
    /// the feature gate is enabled, or issues a feature error if it is not.
    fn impl_trait_in_bindings_ctxt(&self, position: ImplTraitPosition) -> ImplTraitContext {
        if self.tcx.features().impl_trait_in_bindings() {
            ImplTraitContext::InBinding
        } else {
            ImplTraitContext::FeatureGated(position, sym::impl_trait_in_bindings)
        }
    }

    fn lower_local(&mut self, l: &Local) -> &'hir hir::LetStmt<'hir> {
        // Let statements are allowed to have impl trait in bindings.
        let super_ = l.super_;
        let ty = l.ty.as_ref().map(|t| {
            self.lower_ty(t, self.impl_trait_in_bindings_ctxt(ImplTraitPosition::Variable))
        });
        let init = l.kind.init().map(|init| self.lower_expr(init));
        let hir_id = self.lower_node_id(l.id);
        let pat = self.lower_pat(&l.pat);
        let els = if let LocalKind::InitElse(_, els) = &l.kind {
            Some(self.lower_block(els, false))
        } else {
            None
        };
        let span = self.lower_span(l.span);
        let source = hir::LocalSource::Normal;
        self.lower_attrs(hir_id, &l.attrs, l.span);
        self.arena.alloc(hir::LetStmt { hir_id, super_, ty, pat, init, els, span, source })
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
