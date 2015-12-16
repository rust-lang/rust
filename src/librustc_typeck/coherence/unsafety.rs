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
use rustc_front::intravisit;
use rustc_front::hir;

pub fn check(tcx: &ty::ctxt) {
    let mut orphan = UnsafetyChecker { tcx: tcx };
    tcx.map.krate().visit_all_items(&mut orphan);
}

struct UnsafetyChecker<'cx, 'tcx:'cx> {
    tcx: &'cx ty::ctxt<'tcx>
}

impl<'cx, 'tcx, 'v> UnsafetyChecker<'cx, 'tcx> {
    fn check_unsafety_coherence(&mut self, item: &'v hir::Item,
                                unsafety: hir::Unsafety,
                                polarity: hir::ImplPolarity) {
        match self.tcx.impl_trait_ref(self.tcx.map.local_def_id(item.id)) {
            None => {
                // Inherent impl.
                match unsafety {
                    hir::Unsafety::Normal => { /* OK */ }
                    hir::Unsafety::Unsafe => {
                        span_err!(self.tcx.sess, item.span, E0197,
                                  "inherent impls cannot be declared as unsafe");
                    }
                }
            }

            Some(trait_ref) => {
                let trait_def = self.tcx.lookup_trait_def(trait_ref.def_id);
                match (trait_def.unsafety, unsafety, polarity) {
                    (hir::Unsafety::Unsafe,
                     hir::Unsafety::Unsafe, hir::ImplPolarity::Negative) => {
                        span_err!(self.tcx.sess, item.span, E0198,
                                  "negative implementations are not unsafe");
                    }

                    (hir::Unsafety::Normal, hir::Unsafety::Unsafe, _) => {
                        span_err!(self.tcx.sess, item.span, E0199,
                                  "implementing the trait `{}` is not unsafe",
                                  trait_ref);
                    }

                    (hir::Unsafety::Unsafe,
                     hir::Unsafety::Normal, hir::ImplPolarity::Positive) => {
                        span_err!(self.tcx.sess, item.span, E0200,
                                  "the trait `{}` requires an `unsafe impl` declaration",
                                  trait_ref);
                    }

                    (hir::Unsafety::Unsafe,
                     hir::Unsafety::Normal, hir::ImplPolarity::Negative) |
                    (hir::Unsafety::Unsafe,
                     hir::Unsafety::Unsafe, hir::ImplPolarity::Positive) |
                    (hir::Unsafety::Normal, hir::Unsafety::Normal, _) => {
                        /* OK */
                    }
                }
            }
        }
    }
}

impl<'cx, 'tcx,'v> intravisit::Visitor<'v> for UnsafetyChecker<'cx, 'tcx> {
    fn visit_item(&mut self, item: &'v hir::Item) {
        match item.node {
            hir::ItemDefaultImpl(unsafety, _) => {
                self.check_unsafety_coherence(item, unsafety, hir::ImplPolarity::Positive);
            }
            hir::ItemImpl(unsafety, polarity, _, _, _, _) => {
                self.check_unsafety_coherence(item, unsafety, polarity);
            }
            _ => { }
        }
    }
}
