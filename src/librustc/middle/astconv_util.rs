// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * This module contains a simple utility routine
 * used by both `typeck` and `const_eval`.
 * Almost certainly this could (and should) be refactored out of existence.
 */

use middle::def;
use middle::ty::{self, Ty};

use syntax::codemap::Span;
use rustc_front::hir as ast;

pub fn prohibit_type_params(tcx: &ty::ctxt, segments: &[ast::PathSegment]) {
    for segment in segments {
        for typ in segment.parameters.types() {
            span_err!(tcx.sess, typ.span, E0109,
                      "type parameters are not allowed on this type");
            break;
        }
        for lifetime in segment.parameters.lifetimes() {
            span_err!(tcx.sess, lifetime.span, E0110,
                      "lifetime parameters are not allowed on this type");
            break;
        }
        for binding in segment.parameters.bindings() {
            prohibit_projection(tcx, binding.span);
            break;
        }
    }
}

pub fn prohibit_projection(tcx: &ty::ctxt, span: Span)
{
    span_err!(tcx.sess, span, E0229,
              "associated type bindings are not allowed here");
}

pub fn prim_ty_to_ty<'tcx>(tcx: &ty::ctxt<'tcx>,
                           segments: &[ast::PathSegment],
                           nty: ast::PrimTy)
                           -> Ty<'tcx> {
    prohibit_type_params(tcx, segments);
    match nty {
        ast::TyBool => tcx.types.bool,
        ast::TyChar => tcx.types.char,
        ast::TyInt(it) => tcx.mk_mach_int(it),
        ast::TyUint(uit) => tcx.mk_mach_uint(uit),
        ast::TyFloat(ft) => tcx.mk_mach_float(ft),
        ast::TyStr => tcx.mk_str()
    }
}

/// If a type in the AST is a primitive type, return the ty::Ty corresponding
/// to it.
pub fn ast_ty_to_prim_ty<'tcx>(tcx: &ty::ctxt<'tcx>, ast_ty: &ast::Ty)
                               -> Option<Ty<'tcx>> {
    if let ast::TyPath(None, ref path) = ast_ty.node {
        let def = match tcx.def_map.borrow().get(&ast_ty.id) {
            None => {
                tcx.sess.span_bug(ast_ty.span,
                                  &format!("unbound path {:?}", path))
            }
            Some(d) => d.full_def()
        };
        if let def::DefPrimTy(nty) = def {
            Some(prim_ty_to_ty(tcx, &path.segments, nty))
        } else {
            None
        }
    } else {
        None
    }
}
