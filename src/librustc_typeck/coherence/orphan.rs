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

use rustc::traits;
use rustc::ty::{self, TyCtxt};
use rustc::dep_graph::DepNode;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir;

pub fn check<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let mut orphan = OrphanChecker { tcx: tcx };
    tcx.visit_all_item_likes_in_krate(DepNode::CoherenceOrphanCheck, &mut orphan);
}

struct OrphanChecker<'cx, 'tcx: 'cx> {
    tcx: TyCtxt<'cx, 'tcx, 'tcx>,
}

impl<'cx, 'tcx, 'v> ItemLikeVisitor<'v> for OrphanChecker<'cx, 'tcx> {
    /// Checks exactly one impl for orphan rules and other such
    /// restrictions.  In this fn, it can happen that multiple errors
    /// apply to a specific impl, so just return after reporting one
    /// to prevent inundating the user with a bunch of similar error
    /// reports.
    fn visit_item(&mut self, item: &hir::Item) {
        let def_id = self.tcx.hir.local_def_id(item.id);
        match item.node {
            hir::ItemImpl(.., Some(_), _, _) => {
                // "Trait" impl
                debug!("coherence2::orphan check: trait impl {}",
                       self.tcx.hir.node_to_string(item.id));
                let trait_ref = self.tcx.impl_trait_ref(def_id).unwrap();
                let trait_def_id = trait_ref.def_id;
                match traits::orphan_check(self.tcx, def_id) {
                    Ok(()) => {}
                    Err(traits::OrphanCheckErr::NoLocalInputType) => {
                        struct_span_err!(self.tcx.sess,
                                         item.span,
                                         E0117,
                                         "only traits defined in the current crate can be \
                                          implemented for arbitrary types")
                            .span_label(item.span, &format!("impl doesn't use types inside crate"))
                            .note(&format!("the impl does not reference any types defined in \
                                            this crate"))
                            .note("define and implement a trait or new type instead")
                            .emit();
                        return;
                    }
                    Err(traits::OrphanCheckErr::UncoveredTy(param_ty)) => {
                        span_err!(self.tcx.sess,
                                  item.span,
                                  E0210,
                                  "type parameter `{}` must be used as the type parameter for \
                                   some local type (e.g. `MyStruct<T>`); only traits defined in \
                                   the current crate can be implemented for a type parameter",
                                  param_ty);
                        return;
                    }
                }

                // In addition to the above rules, we restrict impls of defaulted traits
                // so that they can only be implemented on structs/enums. To see why this
                // restriction exists, consider the following example (#22978). Imagine
                // that crate A defines a defaulted trait `Foo` and a fn that operates
                // on pairs of types:
                //
                // ```
                // // Crate A
                // trait Foo { }
                // impl Foo for .. { }
                // fn two_foos<A:Foo,B:Foo>(..) {
                //     one_foo::<(A,B)>(..)
                // }
                // fn one_foo<T:Foo>(..) { .. }
                // ```
                //
                // This type-checks fine; in particular the fn
                // `two_foos` is able to conclude that `(A,B):Foo`
                // because `A:Foo` and `B:Foo`.
                //
                // Now imagine that crate B comes along and does the following:
                //
                // ```
                // struct A { }
                // struct B { }
                // impl Foo for A { }
                // impl Foo for B { }
                // impl !Send for (A, B) { }
                // ```
                //
                // This final impl is legal according to the orpan
                // rules, but it invalidates the reasoning from
                // `two_foos` above.
                debug!("trait_ref={:?} trait_def_id={:?} trait_has_default_impl={}",
                       trait_ref,
                       trait_def_id,
                       self.tcx.trait_has_default_impl(trait_def_id));
                if self.tcx.trait_has_default_impl(trait_def_id) &&
                   !trait_def_id.is_local() {
                    let self_ty = trait_ref.self_ty();
                    let opt_self_def_id = match self_ty.sty {
                        ty::TyAdt(self_def, _) => Some(self_def.did),
                        _ => None,
                    };

                    let msg = match opt_self_def_id {
                        // We only want to permit structs/enums, but not *all* structs/enums.
                        // They must be local to the current crate, so that people
                        // can't do `unsafe impl Send for Rc<SomethingLocal>` or
                        // `impl !Send for Box<SomethingLocalAndSend>`.
                        Some(self_def_id) => {
                            if self_def_id.is_local() {
                                None
                            } else {
                                Some(format!("cross-crate traits with a default impl, like `{}`, \
                                              can only be implemented for a struct/enum type \
                                              defined in the current crate",
                                             self.tcx.item_path_str(trait_def_id)))
                            }
                        }
                        _ => {
                            Some(format!("cross-crate traits with a default impl, like `{}`, can \
                                          only be implemented for a struct/enum type, not `{}`",
                                         self.tcx.item_path_str(trait_def_id),
                                         self_ty))
                        }
                    };

                    if let Some(msg) = msg {
                        span_err!(self.tcx.sess, item.span, E0321, "{}", msg);
                        return;
                    }
                }
            }
            hir::ItemDefaultImpl(_, ref item_trait_ref) => {
                // "Trait" impl
                debug!("coherence2::orphan check: default trait impl {}",
                       self.tcx.hir.node_to_string(item.id));
                let trait_ref = self.tcx.impl_trait_ref(def_id).unwrap();
                if !trait_ref.def_id.is_local() {
                    struct_span_err!(self.tcx.sess,
                                     item_trait_ref.path.span,
                                     E0318,
                                     "cannot create default implementations for traits outside \
                                      the crate they're defined in; define a new trait instead")
                        .span_label(item_trait_ref.path.span,
                                    &format!("`{}` trait not defined in this crate",
                            self.tcx.hir.node_to_pretty_string(item_trait_ref.ref_id)))
                        .emit();
                    return;
                }
            }
            _ => {
                // Not an impl
            }
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
    }
}
