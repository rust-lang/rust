// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::mem_categorization::Typer;
use middle::ty;
use util::ppaux::{ty_to_string};
use util::ppaux::UserString;

use syntax::ast::*;
use syntax::codemap::Span;
use syntax::visit::Visitor;
use syntax::visit;

// Kind analysis pass. This pass does some ad-hoc checks that are more
// convenient to do after type checking is complete and all checks are
// known. These are generally related to the builtin bounds `Copy` and
// `Sized`. Note that many of the builtin bound properties that used
// to be checked here are actually checked by trait checking these
// days.

pub struct Context<'a,'tcx:'a> {
    tcx: &'a ty::ctxt<'tcx>,
}

impl<'a, 'tcx, 'v> Visitor<'v> for Context<'a, 'tcx> {
    fn visit_ty(&mut self, t: &Ty) {
        check_ty(self, t);
    }
}

pub fn check_crate(tcx: &ty::ctxt) {
    let mut ctx = Context {
        tcx: tcx,
    };
    visit::walk_crate(&mut ctx, tcx.map.krate());
    tcx.sess.abort_if_errors();
}

fn check_ty(cx: &mut Context, aty: &Ty) {
    match aty.node {
        TyPath(_, _, id) => {
            match cx.tcx.item_substs.borrow().find(&id) {
                None => {}
                Some(ref item_substs) => {
                    let def_map = cx.tcx.def_map.borrow();
                    let did = def_map.get_copy(&id).def_id();
                    let generics = ty::lookup_item_type(cx.tcx, did).generics;
                    for def in generics.types.iter() {
                        let ty = *item_substs.substs.types.get(def.space,
                                                               def.index);
                        check_typaram_bounds(cx, aty.span, ty, def);
                    }
                }
            }
        }
        _ => {}
    }

    visit::walk_ty(cx, aty);
}

// Calls "any_missing" if any bounds were missing.
pub fn check_builtin_bounds(cx: &Context,
                            ty: ty::t,
                            bounds: ty::BuiltinBounds,
                            any_missing: |ty::BuiltinBounds|) {
    let kind = ty::type_contents(cx.tcx, ty);
    let mut missing = ty::empty_builtin_bounds();
    for bound in bounds.iter() {
        if !kind.meets_builtin_bound(cx.tcx, bound) {
            missing.add(bound);
        }
    }
    if !missing.is_empty() {
        any_missing(missing);
    }
}

pub fn check_typaram_bounds(cx: &Context,
                            sp: Span,
                            ty: ty::t,
                            type_param_def: &ty::TypeParameterDef) {
    check_builtin_bounds(cx,
                         ty,
                         type_param_def.bounds.builtin_bounds,
                         |missing| {
        span_err!(cx.tcx.sess, sp, E0144,
                  "instantiating a type parameter with an incompatible type \
                   `{}`, which does not fulfill `{}`",
                   ty_to_string(cx.tcx, ty),
                   missing.user_string(cx.tcx));
    });
}

