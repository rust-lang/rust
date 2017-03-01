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
use traits::Reveal;
use ty::{self, Ty, TyCtxt};
use ty::layout::{LayoutError, Pointer, SizeSkeleton};

use syntax::abi::Abi::RustIntrinsic;
use syntax_pos::Span;
use hir::intravisit::{self, Visitor, NestedVisitorMap};
use hir;

pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let mut visitor = ItemVisitor {
        tcx: tcx
    };
    tcx.visit_all_item_likes_in_krate(DepNode::IntrinsicCheck, &mut visitor.as_deep_visitor());
}

struct ItemVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>
}

struct ExprVisitor<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>
}

/// If the type is `Option<T>`, it will return `T`, otherwise
/// the type itself. Works on most `Option`-like types.
fn unpack_option_like<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                ty: Ty<'tcx>)
                                -> Ty<'tcx> {
    let (def, substs) = match ty.sty {
        ty::TyAdt(def, substs) => (def, substs),
        _ => return ty
    };

    if def.variants.len() == 2 && !def.repr.c && def.repr.int.is_none() {
        let data_idx;

        if def.variants[0].fields.is_empty() {
            data_idx = 1;
        } else if def.variants[1].fields.is_empty() {
            data_idx = 0;
        } else {
            return ty;
        }

        if def.variants[data_idx].fields.len() == 1 {
            return def.variants[data_idx].fields[0].ty(tcx, substs);
        }
    }

    ty
}

impl<'a, 'gcx, 'tcx> ExprVisitor<'a, 'gcx, 'tcx> {
    fn def_id_is_transmute(&self, def_id: DefId) -> bool {
        let intrinsic = match self.infcx.tcx.item_type(def_id).sty {
            ty::TyFnDef(.., bfty) => bfty.abi() == RustIntrinsic,
            _ => return false
        };
        intrinsic && self.infcx.tcx.item_name(def_id) == "transmute"
    }

    fn check_transmute(&self, span: Span, from: Ty<'gcx>, to: Ty<'gcx>) {
        let sk_from = SizeSkeleton::compute(from, self.infcx);
        let sk_to = SizeSkeleton::compute(to, self.infcx);

        // Check for same size using the skeletons.
        if let (Ok(sk_from), Ok(sk_to)) = (sk_from, sk_to) {
            if sk_from.same_size(sk_to) {
                return;
            }

            // Special-case transmutting from `typeof(function)` and
            // `Option<typeof(function)>` to present a clearer error.
            let from = unpack_option_like(self.infcx.tcx.global_tcx(), from);
            match (&from.sty, sk_to) {
                (&ty::TyFnDef(..), SizeSkeleton::Known(size_to))
                        if size_to == Pointer.size(&self.infcx.tcx.data_layout) => {
                    struct_span_err!(self.infcx.tcx.sess, span, E0591,
                                     "`{}` is zero-sized and can't be transmuted to `{}`",
                                     from, to)
                        .span_note(span, &format!("cast with `as` to a pointer instead"))
                        .emit();
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

        struct_span_err!(self.infcx.tcx.sess, span, E0512,
                  "transmute called with differently sized types: \
                   {} ({}) to {} ({})",
                  from, skeleton_string(from, sk_from),
                  to, skeleton_string(to, sk_to))
            .span_label(span,
                &format!("transmuting between {} and {}",
                    skeleton_string(from, sk_from),
                    skeleton_string(to, sk_to)))
            .emit();
    }
}

impl<'a, 'tcx> Visitor<'tcx> for ItemVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_nested_body(&mut self, body_id: hir::BodyId) {
        let body = self.tcx.hir.body(body_id);
        self.tcx.infer_ctxt(body_id, Reveal::All).enter(|infcx| {
            let mut visitor = ExprVisitor {
                infcx: &infcx
            };
            visitor.visit_body(body);
        });
        self.visit_body(body);
    }
}

impl<'a, 'gcx, 'tcx> Visitor<'gcx> for ExprVisitor<'a, 'gcx, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'gcx> {
        NestedVisitorMap::None
    }

    fn visit_expr(&mut self, expr: &'gcx hir::Expr) {
        let def = if let hir::ExprPath(ref qpath) = expr.node {
            self.infcx.tables.borrow().qpath_def(qpath, expr.id)
        } else {
            Def::Err
        };
        match def {
            Def::Fn(did) if self.def_id_is_transmute(did) => {
                let typ = self.infcx.tables.borrow().node_id_to_type(expr.id);
                let typ = self.infcx.tcx.lift_to_global(&typ).unwrap();
                match typ.sty {
                    ty::TyFnDef(.., sig) if sig.abi() == RustIntrinsic => {
                        let from = sig.inputs().skip_binder()[0];
                        let to = *sig.output().skip_binder();
                        self.check_transmute(expr.span, from, to);
                    }
                    _ => {
                        span_bug!(expr.span, "transmute wasn't a bare fn?!");
                    }
                }
            }
            _ => {}
        }

        intravisit::walk_expr(self, expr);
    }
}
