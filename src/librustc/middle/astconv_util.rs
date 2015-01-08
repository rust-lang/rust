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
use syntax::ast;
use util::ppaux::Repr;

pub const NO_REGIONS: uint = 1;
pub const NO_TPS: uint = 2;

pub fn check_path_args(tcx: &ty::ctxt,
                       path: &ast::Path,
                       flags: uint) {
    if (flags & NO_TPS) != 0u {
        if path.segments.iter().any(|s| s.parameters.has_types()) {
            span_err!(tcx.sess, path.span, E0109,
                "type parameters are not allowed on this type");
        }
    }

    if (flags & NO_REGIONS) != 0u {
        if path.segments.iter().any(|s| s.parameters.has_lifetimes()) {
            span_err!(tcx.sess, path.span, E0110,
                "region parameters are not allowed on this type");
        }
    }
}

pub fn ast_ty_to_prim_ty<'tcx>(tcx: &ty::ctxt<'tcx>, ast_ty: &ast::Ty)
                               -> Option<Ty<'tcx>> {
    match ast_ty.node {
        ast::TyPath(ref path, id) => {
            let a_def = match tcx.def_map.borrow().get(&id) {
                None => {
                    tcx.sess.span_bug(ast_ty.span,
                                      &format!("unbound path {}",
                                              path.repr(tcx))[])
                }
                Some(&d) => d
            };
            match a_def {
                def::DefPrimTy(nty) => {
                    match nty {
                        ast::TyBool => {
                            check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                            Some(tcx.types.bool)
                        }
                        ast::TyChar => {
                            check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                            Some(tcx.types.char)
                        }
                        ast::TyInt(it) => {
                            check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                            Some(ty::mk_mach_int(tcx, it))
                        }
                        ast::TyUint(uit) => {
                            check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                            Some(ty::mk_mach_uint(tcx, uit))
                        }
                        ast::TyFloat(ft) => {
                            check_path_args(tcx, path, NO_TPS | NO_REGIONS);
                            Some(ty::mk_mach_float(tcx, ft))
                        }
                        ast::TyStr => {
                            Some(ty::mk_str(tcx))
                        }
                    }
                }
                _ => None
            }
        }
        _ => None
    }
}

