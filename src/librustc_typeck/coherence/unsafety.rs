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

use rustc::ty::TyCtxt;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir::{self, Unsafety};

pub fn check<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let mut unsafety = UnsafetyChecker { tcx: tcx };
    tcx.hir.krate().visit_all_item_likes(&mut unsafety);
}

struct UnsafetyChecker<'cx, 'tcx: 'cx> {
    tcx: TyCtxt<'cx, 'tcx, 'tcx>,
}

impl<'cx, 'tcx, 'v> UnsafetyChecker<'cx, 'tcx> {
    fn check_unsafety_coherence(&mut self,
                                item: &'v hir::Item,
                                impl_generics: Option<&hir::Generics>,
                                unsafety: hir::Unsafety,
                                polarity: hir::ImplPolarity) {
        match self.tcx.impl_trait_ref(self.tcx.hir.local_def_id(item.id)) {
            None => {}

            Some(trait_ref) => {
                let trait_def = self.tcx.trait_def(trait_ref.def_id);
                let unsafe_attr = impl_generics.and_then(|generics| {
                    generics.params.iter().find(|p| p.pure_wrt_drop).map(|_| "may_dangle")
                });
                match (trait_def.unsafety, unsafe_attr, unsafety, polarity) {
                    (Unsafety::Normal, None, Unsafety::Unsafe, hir::ImplPolarity::Positive) => {
                        span_err!(self.tcx.sess,
                                  item.span,
                                  E0199,
                                  "implementing the trait `{}` is not unsafe",
                                  trait_ref);
                    }

                    (Unsafety::Unsafe, _, Unsafety::Normal, hir::ImplPolarity::Positive) => {
                        span_err!(self.tcx.sess,
                                  item.span,
                                  E0200,
                                  "the trait `{}` requires an `unsafe impl` declaration",
                                  trait_ref);
                    }

                    (Unsafety::Normal, Some(attr_name), Unsafety::Normal,
                        hir::ImplPolarity::Positive) =>
                    {
                        span_err!(self.tcx.sess,
                                  item.span,
                                  E0569,
                                  "requires an `unsafe impl` declaration due to `#[{}]` attribute",
                                  attr_name);
                    }

                    (_, _, Unsafety::Unsafe, hir::ImplPolarity::Negative) => {
                        // Reported in AST validation
                        self.tcx.sess.delay_span_bug(item.span, "unsafe negative impl");
                    }
                    (_, _, Unsafety::Normal, hir::ImplPolarity::Negative) |
                    (Unsafety::Unsafe, _, Unsafety::Unsafe, hir::ImplPolarity::Positive) |
                    (Unsafety::Normal, Some(_), Unsafety::Unsafe, hir::ImplPolarity::Positive) |
                    (Unsafety::Normal, None, Unsafety::Normal, _) => {
                        // OK
                    }
                }
            }
        }
    }
}

impl<'cx, 'tcx, 'v> ItemLikeVisitor<'v> for UnsafetyChecker<'cx, 'tcx> {
    fn visit_item(&mut self, item: &'v hir::Item) {
        match item.node {
            hir::ItemImpl(unsafety, polarity, _, ref generics, ..) => {
                self.check_unsafety_coherence(item, Some(generics), unsafety, polarity);
            }
            _ => {}
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
    }
}
