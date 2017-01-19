// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Overlap: No two impls for the same trait are implemented for the
//! same type. Likewise, no two inherent impls for a given type
//! constructor provide a method with the same name.

use hir::def_id::DefId;
use rustc::traits::{self, Reveal};
use rustc::ty::{self, TyCtxt, TypeFoldable};
use syntax::ast;
use rustc::dep_graph::DepNode;
use rustc::hir;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use util::nodemap::DefIdMap;
use lint;

pub fn check<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    let mut overlap = OverlapChecker {
        tcx: tcx,
        default_impls: DefIdMap(),
    };

    // this secondary walk specifically checks for some other cases,
    // like defaulted traits, for which additional overlap rules exist
    tcx.visit_all_item_likes_in_krate(DepNode::CoherenceOverlapCheckSpecial, &mut overlap);
}

struct OverlapChecker<'cx, 'tcx: 'cx> {
    tcx: TyCtxt<'cx, 'tcx, 'tcx>,

    // maps from a trait def-id to an impl id
    default_impls: DefIdMap<ast::NodeId>,
}

impl<'cx, 'tcx> OverlapChecker<'cx, 'tcx> {
    fn check_for_common_items_in_impls(&self, impl1: DefId, impl2: DefId) {
        #[derive(Copy, Clone, PartialEq)]
        enum Namespace {
            Type,
            Value,
        }

        let name_and_namespace = |def_id| {
            let item = self.tcx.associated_item(def_id);
            (item.name, match item.kind {
                ty::AssociatedKind::Type => Namespace::Type,
                ty::AssociatedKind::Const |
                ty::AssociatedKind::Method => Namespace::Value,
            })
        };

        let impl_items1 = self.tcx.associated_item_def_ids(impl1);
        let impl_items2 = self.tcx.associated_item_def_ids(impl2);

        for &item1 in &impl_items1[..] {
            let (name, namespace) = name_and_namespace(item1);

            for &item2 in &impl_items2[..] {
                if (name, namespace) == name_and_namespace(item2) {
                    let msg = format!("duplicate definitions with name `{}`", name);
                    let node_id = self.tcx.map.as_local_node_id(item1).unwrap();
                    self.tcx.sess.add_lint(lint::builtin::OVERLAPPING_INHERENT_IMPLS,
                                           node_id,
                                           self.tcx.span_of_impl(item1).unwrap(),
                                           msg);
                }
            }
        }
    }

    fn check_for_overlapping_inherent_impls(&self, ty_def_id: DefId) {
        let _task = self.tcx.dep_graph.in_task(DepNode::CoherenceOverlapInherentCheck(ty_def_id));

        let inherent_impls = self.tcx.inherent_impls.borrow();
        let impls = match inherent_impls.get(&ty_def_id) {
            Some(impls) => impls,
            None => return,
        };

        for (i, &impl1_def_id) in impls.iter().enumerate() {
            for &impl2_def_id in &impls[(i + 1)..] {
                self.tcx.infer_ctxt((), Reveal::ExactMatch).enter(|infcx| {
                    if traits::overlapping_impls(&infcx, impl1_def_id, impl2_def_id).is_some() {
                        self.check_for_common_items_in_impls(impl1_def_id, impl2_def_id)
                    }
                });
            }
        }
    }
}

impl<'cx, 'tcx, 'v> ItemLikeVisitor<'v> for OverlapChecker<'cx, 'tcx> {
    fn visit_item(&mut self, item: &'v hir::Item) {
        match item.node {
            hir::ItemEnum(..) |
            hir::ItemStruct(..) |
            hir::ItemTrait(..) |
            hir::ItemUnion(..) => {
                let type_def_id = self.tcx.map.local_def_id(item.id);
                self.check_for_overlapping_inherent_impls(type_def_id);
            }

            hir::ItemDefaultImpl(..) => {
                // look for another default impl; note that due to the
                // general orphan/coherence rules, it must always be
                // in this crate.
                let impl_def_id = self.tcx.map.local_def_id(item.id);
                let trait_ref = self.tcx.impl_trait_ref(impl_def_id).unwrap();

                let prev_default_impl = self.default_impls.insert(trait_ref.def_id, item.id);
                if let Some(prev_id) = prev_default_impl {
                    let mut err = struct_span_err!(self.tcx.sess,
                                                   self.tcx.span_of_impl(impl_def_id).unwrap(),
                                                   E0521,
                                                   "redundant default implementations of trait \
                                                    `{}`:",
                                                   trait_ref);
                    err.span_note(self.tcx
                                      .span_of_impl(self.tcx.map.local_def_id(prev_id))
                                      .unwrap(),
                                  "redundant implementation is here:");
                    err.emit();
                }
            }
            hir::ItemImpl(.., Some(_), _, _) => {
                let impl_def_id = self.tcx.map.local_def_id(item.id);
                let trait_ref = self.tcx.impl_trait_ref(impl_def_id).unwrap();
                let trait_def_id = trait_ref.def_id;

                if trait_ref.references_error() {
                    debug!("coherence: skipping impl {:?} with error {:?}",
                           impl_def_id, trait_ref);
                    return
                }

                let _task =
                    self.tcx.dep_graph.in_task(DepNode::CoherenceOverlapCheck(trait_def_id));

                let def = self.tcx.lookup_trait_def(trait_def_id);

                // attempt to insert into the specialization graph
                let insert_result = def.add_impl_for_specialization(self.tcx, impl_def_id);

                // insertion failed due to overlap
                if let Err(overlap) = insert_result {
                    let mut err = struct_span_err!(self.tcx.sess,
                                                   self.tcx.span_of_impl(impl_def_id).unwrap(),
                                                   E0119,
                                                   "conflicting implementations of trait `{}`{}:",
                                                   overlap.trait_desc,
                                                   overlap.self_desc.clone().map_or(String::new(),
                                                                                    |ty| {
                        format!(" for type `{}`", ty)
                    }));

                    match self.tcx.span_of_impl(overlap.with_impl) {
                        Ok(span) => {
                            err.span_label(span, &format!("first implementation here"));
                            err.span_label(self.tcx.span_of_impl(impl_def_id).unwrap(),
                                           &format!("conflicting implementation{}",
                                                    overlap.self_desc
                                                        .map_or(String::new(),
                                                                |ty| format!(" for `{}`", ty))));
                        }
                        Err(cname) => {
                            err.note(&format!("conflicting implementation in crate `{}`", cname));
                        }
                    }

                    err.emit();
                }

                // check for overlap with the automatic `impl Trait for Trait`
                if let ty::TyDynamic(ref data, ..) = trait_ref.self_ty().sty {
                    // This is something like impl Trait1 for Trait2. Illegal
                    // if Trait1 is a supertrait of Trait2 or Trait2 is not object safe.

                    if data.principal().map_or(true, |p| !self.tcx.is_object_safe(p.def_id())) {
                        // This is an error, but it will be reported by wfcheck.  Ignore it here.
                        // This is tested by `coherence-impl-trait-for-trait-object-safe.rs`.
                    } else {
                        let mut supertrait_def_ids =
                            traits::supertrait_def_ids(self.tcx,
                                                       data.principal().unwrap().def_id());
                        if supertrait_def_ids.any(|d| d == trait_def_id) {
                            span_err!(self.tcx.sess,
                                      item.span,
                                      E0371,
                                      "the object type `{}` automatically \
                                       implements the trait `{}`",
                                      trait_ref.self_ty(),
                                      self.tcx.item_path_str(trait_def_id));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
    }
}
