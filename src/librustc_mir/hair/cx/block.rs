// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hair::*;
use hair::cx::Cx;
use hair::cx::to_ref::ToRef;
use rustc::middle::region::{BlockRemainder, CodeExtent};
use rustc::hir;
use syntax::ast;

impl<'tcx> Mirror<'tcx> for &'tcx hir::Block {
    type Output = Block<'tcx>;

    fn make_mirror<'a, 'gcx>(self, cx: &mut Cx<'a, 'gcx, 'tcx>) -> Block<'tcx> {
        // We have to eagerly translate the "spine" of the statements
        // in order to get the lexical scoping correctly.
        let stmts = mirror_stmts(cx, self.id, &*self.stmts);
        let opt_def_id = cx.tcx.hir.opt_local_def_id(self.id);
        let opt_destruction_extent = opt_def_id.and_then(|def_id| {
            cx.tcx.region_maps(def_id).opt_destruction_extent(self.id)
        });
        Block {
            targeted_by_break: self.targeted_by_break,
            extent: CodeExtent::Misc(self.id),
            opt_destruction_extent: opt_destruction_extent,
            span: self.span,
            stmts: stmts,
            expr: self.expr.to_ref(),
        }
    }
}

fn mirror_stmts<'a, 'gcx, 'tcx>(cx: &mut Cx<'a, 'gcx, 'tcx>,
                                block_id: ast::NodeId,
                                stmts: &'tcx [hir::Stmt])
                                -> Vec<StmtRef<'tcx>> {
    let mut result = vec![];
    let opt_def_id = cx.tcx.hir.opt_local_def_id(block_id);
    for (index, stmt) in stmts.iter().enumerate() {
        let opt_dxn_ext = opt_def_id.and_then(|def_id| {
            cx.tcx.region_maps(def_id).opt_destruction_extent(stmt.node.id())
        });
        match stmt.node {
            hir::StmtExpr(ref expr, id) |
            hir::StmtSemi(ref expr, id) => {
                result.push(StmtRef::Mirror(Box::new(Stmt {
                    span: stmt.span,
                    kind: StmtKind::Expr {
                        scope: CodeExtent::Misc(id),
                        expr: expr.to_ref(),
                    },
                    opt_destruction_extent: opt_dxn_ext,
                })))
            }
            hir::StmtDecl(ref decl, id) => {
                match decl.node {
                    hir::DeclItem(..) => {
                        // ignore for purposes of the MIR
                    }
                    hir::DeclLocal(ref local) => {
                        let remainder_extent = CodeExtent::Remainder(BlockRemainder {
                            block: block_id,
                            first_statement_index: index as u32,
                        });

                        let pattern = Pattern::from_hir(cx.tcx.global_tcx(),
                                                        cx.param_env.and(cx.identity_substs),
                                                        cx.tables(),
                                                        &local.pat);
                        result.push(StmtRef::Mirror(Box::new(Stmt {
                            span: stmt.span,
                            kind: StmtKind::Let {
                                remainder_scope: remainder_extent,
                                init_scope: CodeExtent::Misc(id),
                                pattern: pattern,
                                initializer: local.init.to_ref(),
                            },
                            opt_destruction_extent: opt_dxn_ext,
                        })));
                    }
                }
            }
        }
    }
    return result;
}

pub fn to_expr_ref<'a, 'gcx, 'tcx>(cx: &mut Cx<'a, 'gcx, 'tcx>,
                                   block: &'tcx hir::Block)
                                   -> ExprRef<'tcx> {
    let block_ty = cx.tables().node_id_to_type(block.hir_id);
    let temp_lifetime = cx.region_maps.temporary_scope(block.id);
    let expr = Expr {
        ty: block_ty,
        temp_lifetime: temp_lifetime,
        span: block.span,
        kind: ExprKind::Block { body: block },
    };
    expr.to_ref()
}
