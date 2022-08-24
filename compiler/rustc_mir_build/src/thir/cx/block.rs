use crate::thir::cx::Cx;

use rustc_hir as hir;
use rustc_middle::middle::region;
use rustc_middle::thir::*;
use rustc_middle::ty;

use rustc_index::vec::Idx;
use rustc_middle::ty::CanonicalUserTypeAnnotation;

impl<'tcx> Cx<'tcx> {
    pub(crate) fn mirror_block(&mut self, block: &'tcx hir::Block<'tcx>) -> BlockId {
        // We have to eagerly lower the "spine" of the statements
        // in order to get the lexical scoping correctly.
        let stmts = self.mirror_stmts(block.hir_id.local_id, block.stmts);
        let opt_destruction_scope =
            self.region_scope_tree.opt_destruction_scope(block.hir_id.local_id);
        let block = Block {
            targeted_by_break: block.targeted_by_break,
            region_scope: region::Scope {
                id: block.hir_id.local_id,
                data: region::ScopeData::Node,
            },
            opt_destruction_scope,
            span: block.span,
            stmts,
            expr: block.expr.map(|expr| self.mirror_expr(expr)),
            safety_mode: match block.rules {
                hir::BlockCheckMode::DefaultBlock => BlockSafety::Safe,
                hir::BlockCheckMode::UnsafeBlock(hir::UnsafeSource::CompilerGenerated) => {
                    BlockSafety::BuiltinUnsafe
                }
                hir::BlockCheckMode::UnsafeBlock(hir::UnsafeSource::UserProvided) => {
                    BlockSafety::ExplicitUnsafe(block.hir_id)
                }
            },
        };

        self.thir.blocks.push(block)
    }

    fn mirror_stmts(
        &mut self,
        block_id: hir::ItemLocalId,
        stmts: &'tcx [hir::Stmt<'tcx>],
    ) -> Box<[StmtId]> {
        stmts
            .iter()
            .enumerate()
            .filter_map(|(index, stmt)| {
                let hir_id = stmt.hir_id;
                let opt_dxn_ext = self.region_scope_tree.opt_destruction_scope(hir_id.local_id);
                match stmt.kind {
                    hir::StmtKind::Expr(ref expr) | hir::StmtKind::Semi(ref expr) => {
                        let stmt = Stmt {
                            kind: StmtKind::Expr {
                                scope: region::Scope {
                                    id: hir_id.local_id,
                                    data: region::ScopeData::Node,
                                },
                                expr: self.mirror_expr(expr),
                            },
                            opt_destruction_scope: opt_dxn_ext,
                        };
                        Some(self.thir.stmts.push(stmt))
                    }
                    hir::StmtKind::Item(..) => {
                        // ignore for purposes of the MIR
                        None
                    }
                    hir::StmtKind::Local(ref local) => {
                        let remainder_scope = region::Scope {
                            id: block_id,
                            data: region::ScopeData::Remainder(region::FirstStatementIndex::new(
                                index,
                            )),
                        };

                        let else_block = local.els.map(|els| self.mirror_block(els));

                        let mut pattern = self.pattern_from_hir(local.pat);
                        debug!(?pattern);

                        if let Some(ty) = &local.ty {
                            if let Some(&user_ty) =
                                self.typeck_results.user_provided_types().get(ty.hir_id)
                            {
                                debug!("mirror_stmts: user_ty={:?}", user_ty);
                                let annotation = CanonicalUserTypeAnnotation {
                                    user_ty,
                                    span: ty.span,
                                    inferred_ty: self.typeck_results.node_type(ty.hir_id),
                                };
                                pattern = Pat {
                                    ty: pattern.ty,
                                    span: pattern.span,
                                    kind: Box::new(PatKind::AscribeUserType {
                                        ascription: Ascription {
                                            annotation,
                                            variance: ty::Variance::Covariant,
                                        },
                                        subpattern: pattern,
                                    }),
                                };
                            }
                        }

                        let stmt = Stmt {
                            kind: StmtKind::Let {
                                remainder_scope,
                                init_scope: region::Scope {
                                    id: hir_id.local_id,
                                    data: region::ScopeData::Node,
                                },
                                pattern,
                                initializer: local.init.map(|init| self.mirror_expr(init)),
                                else_block,
                                lint_level: LintLevel::Explicit(local.hir_id),
                            },
                            opt_destruction_scope: opt_dxn_ext,
                        };
                        Some(self.thir.stmts.push(stmt))
                    }
                }
            })
            .collect()
    }
}
