// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use metadata::csearch;
use middle::def::DefFn;
use middle::subst::Subst;
use middle::ty::{TransmuteRestriction, ctxt, ty_bare_fn};
use middle::ty;

use syntax::abi::RustIntrinsic;
use syntax::ast::DefId;
use syntax::ast;
use syntax::ast_map::NodeForeignItem;
use syntax::codemap::Span;
use syntax::parse::token;
use syntax::visit::Visitor;
use syntax::visit;

fn type_size_is_affected_by_type_parameters(tcx: &ty::ctxt, typ: ty::t)
                                            -> bool {
    let mut result = false;
    ty::maybe_walk_ty(typ, |typ| {
        match ty::get(typ).sty {
            ty::ty_uniq(_) | ty::ty_ptr(_) | ty::ty_rptr(..) |
            ty::ty_bare_fn(..) | ty::ty_closure(..) => {
                false
            }
            ty::ty_param(_) => {
                result = true;
                // No need to continue; we now know the result.
                false
            }
            ty::ty_enum(did, ref substs) => {
                for enum_variant in (*ty::enum_variants(tcx, did)).iter() {
                    for argument_type in enum_variant.args.iter() {
                        let argument_type = argument_type.subst(tcx, substs);
                        result = result ||
                            type_size_is_affected_by_type_parameters(
                                tcx,
                                argument_type);
                    }
                }

                // Don't traverse substitutions.
                false
            }
            ty::ty_struct(did, ref substs) => {
                for field in ty::struct_fields(tcx, did, substs).iter() {
                    result = result ||
                        type_size_is_affected_by_type_parameters(tcx,
                                                                 field.mt.ty);
                }

                // Don't traverse substitutions.
                false
            }
            _ => true,
        }
    });
    result
}

struct IntrinsicCheckingVisitor<'a, 'tcx: 'a> {
    tcx: &'a ctxt<'tcx>,
}

impl<'a, 'tcx> IntrinsicCheckingVisitor<'a, 'tcx> {
    fn def_id_is_transmute(&self, def_id: DefId) -> bool {
        let intrinsic = match ty::get(ty::lookup_item_type(self.tcx, def_id).ty).sty {
            ty::ty_bare_fn(ref bfty) => bfty.abi == RustIntrinsic,
            _ => return false
        };
        if def_id.krate == ast::LOCAL_CRATE {
            match self.tcx.map.get(def_id.node) {
                NodeForeignItem(ref item) if intrinsic => {
                    token::get_ident(item.ident) ==
                        token::intern_and_get_ident("transmute")
                }
                _ => false,
            }
        } else {
            match csearch::get_item_path(self.tcx, def_id).last() {
                Some(ref last) if intrinsic => {
                    token::get_name(last.name()) ==
                        token::intern_and_get_ident("transmute")
                }
                _ => false,
            }
        }
    }

    fn check_transmute(&self, span: Span, from: ty::t, to: ty::t, id: ast::NodeId) {
        if type_size_is_affected_by_type_parameters(self.tcx, from) {
            span_err!(self.tcx.sess, span, E0139,
                      "cannot transmute from a type that contains type parameters");
        }
        if type_size_is_affected_by_type_parameters(self.tcx, to) {
            span_err!(self.tcx.sess, span, E0140,
                      "cannot transmute to a type that contains type parameters");
        }

        let restriction = TransmuteRestriction {
            span: span,
            from: from,
            to: to,
            id: id,
        };
        self.tcx.transmute_restrictions.borrow_mut().push(restriction);
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for IntrinsicCheckingVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &ast::Expr) {
        match expr.node {
            ast::ExprPath(..) => {
                match ty::resolve_expr(self.tcx, expr) {
                    DefFn(did, _, _) if self.def_id_is_transmute(did) => {
                        let typ = ty::node_id_to_type(self.tcx, expr.id);
                        match ty::get(typ).sty {
                            ty_bare_fn(ref bare_fn_ty)
                                    if bare_fn_ty.abi == RustIntrinsic => {
                                let from = *bare_fn_ty.sig.inputs.get(0);
                                let to = bare_fn_ty.sig.output;
                                self.check_transmute(expr.span, from, to, expr.id);
                            }
                            _ => {
                                self.tcx
                                    .sess
                                    .span_bug(expr.span,
                                              "transmute wasn't a bare fn?!");
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        visit::walk_expr(self, expr);
    }
}

pub fn check_crate(tcx: &ctxt) {
    visit::walk_crate(&mut IntrinsicCheckingVisitor { tcx: tcx },
                      tcx.map.krate());
}

