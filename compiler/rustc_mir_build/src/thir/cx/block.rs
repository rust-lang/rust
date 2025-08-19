use rustc_hir as hir;
use rustc_index::Idx;
use rustc_middle::middle::region;
use rustc_middle::thir::*;
use rustc_middle::ty;
use rustc_middle::ty::CanonicalUserTypeAnnotation;
use tracing::debug;

use crate::thir::cx::ThirBuildCx;

impl<'tcx> ThirBuildCx<'tcx> {
    pub(crate) fn mirror_block(&mut self, block: &'tcx hir::Block<'tcx>) -> BlockId {
        // We have to eagerly lower the "spine" of the statements
        // in order to get the lexical scoping correctly.
        let stmts = self.mirror_stmts(block.hir_id.local_id, block.stmts);
        let block = Block {
            targeted_by_break: block.targeted_by_break,
            region_scope: region::Scope {
                local_id: block.hir_id.local_id,
                data: region::ScopeData::Node,
            },
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
                match stmt.kind {
                    hir::StmtKind::Expr(expr) | hir::StmtKind::Semi(expr) => {
                        let stmt = Stmt {
                            kind: StmtKind::Expr {
                                scope: region::Scope {
                                    local_id: hir_id.local_id,
                                    data: region::ScopeData::Node,
                                },
                                expr: self.mirror_expr(expr),
                            },
                        };
                        Some(self.thir.stmts.push(stmt))
                    }
                    hir::StmtKind::Item(..) => {
                        // ignore for purposes of the MIR
                        None
                    }
                    hir::StmtKind::Let(local) => {
                        let remainder_scope = region::Scope {
                            local_id: block_id,
                            data: region::ScopeData::Remainder(region::FirstStatementIndex::new(
                                index,
                            )),
                        };

                        let else_block = local.els.map(|els| self.mirror_block(els));

                        let mut pattern = self.pattern_from_hir(local.pat);
                        debug!(?pattern);

                        if let Some(ty) = &local.ty
                            && let Some(&user_ty) =
                                self.typeck_results.user_provided_types().get(ty.hir_id)
                        {
                            debug!("mirror_stmts: user_ty={:?}", user_ty);
                            let annotation = CanonicalUserTypeAnnotation {
                                user_ty: Box::new(user_ty),
                                span: ty.span,
                                inferred_ty: self.typeck_results.node_type(ty.hir_id),
                            };
                            pattern = Box::new(Pat {
                                ty: pattern.ty,
                                span: pattern.span,
                                kind: PatKind::AscribeUserType {
                                    ascription: Ascription { annotation, variance: ty::Covariant },
                                    subpattern: pattern,
                                },
                            });
                        }

                        let span = match local.init {
                            Some(init) => local.span.with_hi(init.span.hi()),
                            None => local.span,
                        };
                        let stmt = Stmt {
                            kind: StmtKind::Let {
                                remainder_scope,
                                init_scope: region::Scope {
                                    local_id: hir_id.local_id,
                                    data: region::ScopeData::Node,
                                },
                                pattern,
                                initializer: local.init.map(|init| self.mirror_expr(init)),
                                else_block,
                                lint_level: LintLevel::Explicit(local.hir_id),
                                span,
                            },
                        };
                        Some(self.thir.stmts.push(stmt))
                    }
                }
            })
            .collect()
    }
}
