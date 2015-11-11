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
use rustc::middle::region::{BlockRemainder, CodeExtentData};
use rustc_front::hir;
use syntax::ast;
use syntax::ptr::P;

impl<'tcx> Mirror<'tcx> for &'tcx hir::Block {
    type Output = Block<'tcx>;

    fn make_mirror<'a>(self, cx: &mut Cx<'a, 'tcx>) -> Block<'tcx> {
        // We have to eagerly translate the "spine" of the statements
        // in order to get the lexical scoping correctly.
        let stmts = mirror_stmts(cx, self.id, self.stmts.iter().enumerate());
        Block {
            extent: cx.tcx.region_maps.node_extent(self.id),
            span: self.span,
            stmts: stmts,
            expr: self.expr.to_ref(),
        }
    }
}

fn mirror_stmts<'a,'tcx:'a,STMTS>(cx: &mut Cx<'a,'tcx>,
                                  block_id: ast::NodeId,
                                  mut stmts: STMTS)
                                  -> Vec<StmtRef<'tcx>>
    where STMTS: Iterator<Item=(usize, &'tcx P<hir::Stmt>)>
{
    let mut result = vec![];
    while let Some((index, stmt)) = stmts.next() {
        match stmt.node {
            hir::StmtExpr(ref expr, id) | hir::StmtSemi(ref expr, id) =>
                result.push(
                    StmtRef::Mirror(
                        Box::new(Stmt { span: stmt.span,
                                        kind: StmtKind::Expr {
                                            scope: cx.tcx.region_maps.node_extent(id),
                                            expr: expr.to_ref() } }))),

            hir::StmtDecl(ref decl, id) => {
                match decl.node {
                    hir::DeclItem(..) => { /* ignore for purposes of the MIR */ }
                    hir::DeclLocal(ref local) => {
                        let remainder_extent = CodeExtentData::Remainder(BlockRemainder {
                            block: block_id,
                            first_statement_index: index as u32,
                        });
                        let remainder_extent =
                            cx.tcx.region_maps.lookup_code_extent(remainder_extent);

                        // pull in all following statements, since
                        // they are within the scope of this let:
                        let following_stmts = mirror_stmts(cx, block_id, stmts);

                        let pattern = cx.irrefutable_pat(&local.pat);
                        result.push(StmtRef::Mirror(Box::new(Stmt {
                            span: stmt.span,
                            kind: StmtKind::Let {
                                remainder_scope: remainder_extent,
                                init_scope: cx.tcx.region_maps.node_extent(id),
                                pattern: pattern,
                                initializer: local.init.to_ref(),
                                stmts: following_stmts,
                            },
                        })));

                        return result;
                    }
                }
            }
        }
    }
    return result;
}

pub fn to_expr_ref<'a, 'tcx: 'a>(cx: &mut Cx<'a, 'tcx>, block: &'tcx hir::Block) -> ExprRef<'tcx> {
    let block_ty = cx.tcx.node_id_to_type(block.id);
    let temp_lifetime = cx.tcx.region_maps.temporary_scope(block.id);
    let expr = Expr {
        ty: block_ty,
        temp_lifetime: temp_lifetime,
        span: block.span,
        kind: ExprKind::Block { body: block },
    };
    expr.to_ref()
}
