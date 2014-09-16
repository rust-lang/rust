// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Orphan checker: every impl either implements a trait defined in this
 * crate or pertains to a type defined in this crate.
 */

use middle::traits;
use middle::ty;
use syntax::ast::{Item, ItemImpl};
use syntax::ast;
use syntax::ast_util;
use syntax::visit;
use util::ppaux::Repr;

pub fn check(tcx: &ty::ctxt) {
    let mut orphan = OrphanChecker { tcx: tcx };
    visit::walk_crate(&mut orphan, tcx.map.krate());
}

struct OrphanChecker<'cx, 'tcx:'cx> {
    tcx: &'cx ty::ctxt<'tcx>
}

impl<'cx, 'tcx,'v> visit::Visitor<'v> for OrphanChecker<'cx, 'tcx> {
    fn visit_item(&mut self, item: &'v ast::Item) {
        let def_id = ast_util::local_def(item.id);
        match item.node {
            ast::ItemImpl(_, None, _, _) => {
                // For inherent impls, self type must be a nominal type
                // defined in this crate.
                debug!("coherence2::orphan check: inherent impl {}", item.repr(self.tcx));
                let self_ty = ty::lookup_item_type(self.tcx, def_id).ty;
                match ty::get(self_ty).sty {
                    ty::ty_enum(def_id, _) |
                    ty::ty_struct(def_id, _) => {
                        if def_id.krate != ast::LOCAL_CRATE {
                            span_err!(self.tcx.sess, item.span, E0116,
                                      "cannot associate methods with a type outside the \
                                      crate the type is defined in; define and implement \
                                      a trait or new type instead");
                        }
                    }
                    _ => {
                        span_err!(self.tcx.sess, item.span, E0118,
                                  "no base type found for inherent implementation; \
                                   implement a trait or new type instead");
                    }
                }
            }
            ast::ItemImpl(_, Some(_), _, _) => {
                // "Trait" impl
                debug!("coherence2::orphan check: trait impl {}", item.repr(self.tcx));
                if traits::is_orphan_impl(self.tcx, def_id) {
                    span_err!(self.tcx.sess, item.span, E0117,
                              "cannot provide an extension implementation \
                               where both trait and type are not defined in this crate");
                }
            }
            _ => {
                // Not an impl
            }
        }

        visit::walk_item(self, item);
    }
}
