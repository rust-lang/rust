// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unsafety checker: every impl either implements a trait defined in this
//! crate or pertains to a type defined in this crate.

use middle::ty;
use syntax::ast::{Item, ItemImpl};
use syntax::ast;
use syntax::ast_util;
use syntax::visit;
use util::ppaux::UserString;

pub fn check(tcx: &ty::ctxt) {
    let mut orphan = UnsafetyChecker { tcx: tcx };
    visit::walk_crate(&mut orphan, tcx.map.krate());
}

struct UnsafetyChecker<'cx, 'tcx:'cx> {
    tcx: &'cx ty::ctxt<'tcx>
}

impl<'cx, 'tcx,'v> visit::Visitor<'v> for UnsafetyChecker<'cx, 'tcx> {
    fn visit_item(&mut self, item: &'v ast::Item) {
        match item.node {
            ast::ItemImpl(unsafety, _, _, _, _) => {
                match ty::impl_trait_ref(self.tcx, ast_util::local_def(item.id)) {
                    None => {
                        // Inherent impl.
                        match unsafety {
                            ast::Unsafety::Normal => { /* OK */ }
                            ast::Unsafety::Unsafe => {
                                self.tcx.sess.span_err(
                                    item.span,
                                    "inherent impls cannot be declared as unsafe");
                            }
                        }
                    }

                    Some(trait_ref) => {
                        let trait_def = ty::lookup_trait_def(self.tcx, trait_ref.def_id());
                        match (trait_def.unsafety, unsafety) {
                            (ast::Unsafety::Normal, ast::Unsafety::Unsafe) => {
                                self.tcx.sess.span_err(
                                    item.span,
                                    format!("implementing the trait `{}` is not unsafe",
                                            trait_ref.user_string(self.tcx)).as_slice());
                            }

                            (ast::Unsafety::Unsafe, ast::Unsafety::Normal) => {
                                self.tcx.sess.span_err(
                                    item.span,
                                    format!("the trait `{}` requires an `unsafe impl` declaration",
                                            trait_ref.user_string(self.tcx)).as_slice());
                            }

                            (ast::Unsafety::Unsafe, ast::Unsafety::Unsafe) |
                            (ast::Unsafety::Normal, ast::Unsafety::Normal) => {
                                /* OK */
                            }
                        }
                    }
                }
            }
            _ => { }
        }

        visit::walk_item(self, item);
    }
}
