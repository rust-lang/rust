// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use dep_graph::DepNode;
use hir::def::Def;
use hir::def_id::DefId;
use infer::InferCtxt;
use traits::ProjectionMode;
use ty::{self, Ty, TyCtxt};
use ty::layout::{LayoutError, Pointer, SizeSkeleton};

use syntax::abi::Abi::RustIntrinsic;
use syntax::ast;
use syntax_pos::Span;
use hir::intravisit::{self, Visitor, FnKind};
use hir;

pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let mut visitor = ItemVisitor {
        tcx: tcx
    };
    tcx.visit_all_items_in_krate(DepNode::IntrinsicCheck, &mut visitor);
}

struct ItemVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>
}

impl<'a, 'tcx> ItemVisitor<'a, 'tcx> {
    fn visit_const(&mut self, item_id: ast::NodeId, expr: &hir::Expr) {
        let param_env = ty::ParameterEnvironment::for_item(self.tcx, item_id);
        self.tcx.infer_ctxt(None, Some(param_env), ProjectionMode::Any).enter(|infcx| {
            let mut visitor = ExprVisitor {
                infcx: &infcx
            };
            visitor.visit_expr(expr);
        });
    }
}

struct ExprVisitor<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>
}

impl<'a, 'gcx, 'tcx> ExprVisitor<'a, 'gcx, 'tcx> {
    fn def_id_is_transmute(&self, def_id: DefId) -> bool {
        let intrinsic = match self.infcx.tcx.lookup_item_type(def_id).ty.sty {
            ty::TyFnDef(_, _, ref bfty) => bfty.abi == RustIntrinsic,
            _ => return false
        };
        intrinsic && self.infcx.tcx.item_name(def_id).as_str() == "transmute"
    }

    fn check_transmute(&self, span: Span, from: Ty<'gcx>, to: Ty<'gcx>, id: ast::NodeId) {
        let sk_from = SizeSkeleton::compute(from, self.infcx);
        let sk_to = SizeSkeleton::compute(to, self.infcx);

        // Check for same size using the skeletons.
        if let (Ok(sk_from), Ok(sk_to)) = (sk_from, sk_to) {
            if sk_from.same_size(sk_to) {
                return;
            }

            match (&from.sty, sk_to) {
                (&ty::TyFnDef(..), SizeSkeleton::Known(size_to))
                        if size_to == Pointer.size(&self.infcx.tcx.data_layout) => {
                    // FIXME #19925 Remove this warning after a release cycle.
                    let msg = format!("`{}` is now zero-sized and has to be cast \
                                       to a pointer before transmuting to `{}`",
                                      from, to);
                    self.infcx.tcx.sess.add_lint(
                        ::lint::builtin::TRANSMUTE_FROM_FN_ITEM_TYPES, id, span, msg);
                    return;
                }
                _ => {}
            }
        }

        // Try to display a sensible error with as much information as possible.
        let skeleton_string = |ty: Ty<'gcx>, sk| {
            match sk {
                Ok(SizeSkeleton::Known(size)) => {
                    format!("{} bits", size.bits())
                }
                Ok(SizeSkeleton::Pointer { tail, .. }) => {
                    format!("pointer to {}", tail)
                }
                Err(LayoutError::Unknown(bad)) => {
                    if bad == ty {
                        format!("size can vary")
                    } else {
                        format!("size can vary because of {}", bad)
                    }
                }
                Err(err) => err.to_string()
            }
        };

        span_err!(self.infcx.tcx.sess, span, E0512,
                  "transmute called with differently sized types: \
                   {} ({}) to {} ({})",
                  from, skeleton_string(from, sk_from),
                  to, skeleton_string(to, sk_to));
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for ItemVisitor<'a, 'tcx> {
    // const, static and N in [T; N].
    fn visit_expr(&mut self, expr: &hir::Expr) {
        self.tcx.infer_ctxt(None, None, ProjectionMode::Any).enter(|infcx| {
            let mut visitor = ExprVisitor {
                infcx: &infcx
            };
            visitor.visit_expr(expr);
        });
    }

    fn visit_trait_item(&mut self, item: &hir::TraitItem) {
        if let hir::ConstTraitItem(_, Some(ref expr)) = item.node {
            self.visit_const(item.id, expr);
        } else {
            intravisit::walk_trait_item(self, item);
        }
    }

    fn visit_impl_item(&mut self, item: &hir::ImplItem) {
        if let hir::ImplItemKind::Const(_, ref expr) = item.node {
            self.visit_const(item.id, expr);
        } else {
            intravisit::walk_impl_item(self, item);
        }
    }

    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v hir::FnDecl,
                b: &'v hir::Block, s: Span, id: ast::NodeId) {
        if let FnKind::Closure(..) = fk {
            span_bug!(s, "intrinsicck: closure outside of function")
        }
        let param_env = ty::ParameterEnvironment::for_item(self.tcx, id);
        self.tcx.infer_ctxt(None, Some(param_env), ProjectionMode::Any).enter(|infcx| {
            let mut visitor = ExprVisitor {
                infcx: &infcx
            };
            visitor.visit_fn(fk, fd, b, s, id);
        });
    }
}

impl<'a, 'gcx, 'tcx, 'v> Visitor<'v> for ExprVisitor<'a, 'gcx, 'tcx> {
    fn visit_expr(&mut self, expr: &hir::Expr) {
        if let hir::ExprPath(..) = expr.node {
            match self.infcx.tcx.expect_def(expr.id) {
                Def::Fn(did) if self.def_id_is_transmute(did) => {
                    let typ = self.infcx.tcx.node_id_to_type(expr.id);
                    match typ.sty {
                        ty::TyFnDef(_, _, ref bare_fn_ty) if bare_fn_ty.abi == RustIntrinsic => {
                            if let ty::FnConverging(to) = bare_fn_ty.sig.0.output {
                                let from = bare_fn_ty.sig.0.inputs[0];
                                self.check_transmute(expr.span, from, to, expr.id);
                            }
                        }
                        _ => {
                            span_bug!(expr.span, "transmute wasn't a bare fn?!");
                        }
                    }
                }
                _ => {}
            }
        }

        intravisit::walk_expr(self, expr);
    }
}
