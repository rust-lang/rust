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

use hir::def::Def;
use ty::{Ty, TyCtxt};

use syntax_pos::Span;
use hir as ast;

impl<'a, 'gcx, 'tcx> TyCtxt<'a, 'gcx, 'tcx> {
    pub fn prohibit_type_params(self, segments: &[ast::PathSegment]) {
        for segment in segments {
            for typ in segment.parameters.types() {
                span_err!(self.sess, typ.span, E0109,
                          "type parameters are not allowed on this type");
                break;
            }
            for lifetime in segment.parameters.lifetimes() {
                span_err!(self.sess, lifetime.span, E0110,
                          "lifetime parameters are not allowed on this type");
                break;
            }
            for binding in segment.parameters.bindings() {
                self.prohibit_projection(binding.span);
                break;
            }
        }
    }

    pub fn prohibit_projection(self, span: Span)
    {
        span_err!(self.sess, span, E0229,
                  "associated type bindings are not allowed here");
    }

    pub fn prim_ty_to_ty(self,
                         segments: &[ast::PathSegment],
                         nty: ast::PrimTy)
                         -> Ty<'tcx> {
        self.prohibit_type_params(segments);
        match nty {
            ast::TyBool => self.types.bool,
            ast::TyChar => self.types.char,
            ast::TyInt(it) => self.mk_mach_int(it),
            ast::TyUint(uit) => self.mk_mach_uint(uit),
            ast::TyFloat(ft) => self.mk_mach_float(ft),
            ast::TyStr => self.mk_str()
        }
    }

    /// If a type in the AST is a primitive type, return the ty::Ty corresponding
    /// to it.
    pub fn ast_ty_to_prim_ty(self, ast_ty: &ast::Ty) -> Option<Ty<'tcx>> {
        if let ast::TyPath(None, ref path) = ast_ty.node {
            if let Def::PrimTy(nty) = self.expect_def(ast_ty.id) {
                Some(self.prim_ty_to_ty(&path.segments, nty))
            } else {
                None
            }
        } else {
            None
        }
    }
}
