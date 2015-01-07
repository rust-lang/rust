// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Orphan checker: every impl either implements a trait defined in this
//! crate or pertains to a type defined in this crate.

use middle::traits;
use middle::ty;
use syntax::ast::{Item, ItemImpl};
use syntax::ast;
use syntax::ast_util;
use syntax::codemap::Span;
use syntax::visit;
use util::ppaux::{Repr, UserString};

pub fn check(tcx: &ty::ctxt) {
    let mut orphan = OrphanChecker { tcx: tcx };
    visit::walk_crate(&mut orphan, tcx.map.krate());
}

struct OrphanChecker<'cx, 'tcx:'cx> {
    tcx: &'cx ty::ctxt<'tcx>
}

impl<'cx, 'tcx> OrphanChecker<'cx, 'tcx> {
    fn check_def_id(&self, span: Span, def_id: ast::DefId) {
        if def_id.krate != ast::LOCAL_CRATE {
            span_err!(self.tcx.sess, span, E0116,
                      "cannot associate methods with a type outside the \
                       crate the type is defined in; define and implement \
                       a trait or new type instead");
        }
    }
}

impl<'cx, 'tcx,'v> visit::Visitor<'v> for OrphanChecker<'cx, 'tcx> {
    fn visit_item(&mut self, item: &'v ast::Item) {
        let def_id = ast_util::local_def(item.id);
        match item.node {
            ast::ItemImpl(_, _, _, None, _, _) => {
                // For inherent impls, self type must be a nominal type
                // defined in this crate.
                debug!("coherence2::orphan check: inherent impl {}", item.repr(self.tcx));
                let self_ty = ty::lookup_item_type(self.tcx, def_id).ty;
                match self_ty.sty {
                    ty::ty_enum(def_id, _) |
                    ty::ty_struct(def_id, _) => {
                        self.check_def_id(item.span, def_id);
                    }
                    ty::ty_trait(ref data) => {
                        self.check_def_id(item.span, data.principal_def_id());
                    }
                    ty::ty_uniq(..) => {
                        self.check_def_id(item.span,
                                          self.tcx.lang_items.owned_box()
                                              .unwrap());
                    }
                    _ => {
                        span_err!(self.tcx.sess, item.span, E0118,
                                  "no base type found for inherent implementation; \
                                   implement a trait or new type instead");
                    }
                }
            }
            ast::ItemImpl(_, _, _, Some(_), _, _) => {
                // "Trait" impl
                debug!("coherence2::orphan check: trait impl {}", item.repr(self.tcx));
                let trait_def_id = ty::impl_trait_ref(self.tcx, def_id).unwrap().def_id;
                match traits::orphan_check(self.tcx, def_id) {
                    Ok(()) => { }
                    Err(traits::OrphanCheckErr::NoLocalInputType) => {
                        if !ty::has_attr(self.tcx, trait_def_id, "old_orphan_check") {
                            let self_ty = ty::lookup_item_type(self.tcx, def_id).ty;
                            span_err!(
                                self.tcx.sess, item.span, E0117,
                                "the type `{}` does not reference any \
                                 types defined in this crate; \
                                 only traits defined in the current crate can be \
                                 implemented for arbitrary types",
                                self_ty.user_string(self.tcx));
                        }
                    }
                    Err(traits::OrphanCheckErr::UncoveredTy(param_ty)) => {
                        if !ty::has_attr(self.tcx, trait_def_id, "old_orphan_check") {
                            self.tcx.sess.span_err(
                                item.span,
                                format!(
                                    "type parameter `{}` is not constrained by any local type; \
                                     only traits defined in the current crate can be implemented \
                                     for a type parameter",
                                    param_ty.user_string(self.tcx)).as_slice());
                            self.tcx.sess.span_note(
                                item.span,
                                format!("for a limited time, you can add \
                                         `#![feature(old_orphan_check)]` to your crate \
                                         to disable this rule").as_slice());
                        }
                    }
                }
            }
            _ => {
                // Not an impl
            }
        }

        visit::walk_item(self, item);
    }
}
