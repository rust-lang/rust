// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::dep_graph::DepNode;
use rustc::hir::def_id::DefId;
use rustc::hir;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::lint;
use rustc::traits::{self, Reveal};
use rustc::ty::{self, TyCtxt};

use syntax::ast;
use syntax_pos::Span;

struct InherentCollect<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>
}

impl<'a, 'tcx, 'v> ItemLikeVisitor<'v> for InherentCollect<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        let (unsafety, ty) = match item.node {
            hir::ItemImpl(unsafety, .., None, ref ty, _) => (unsafety, ty),
            _ => return
        };

        match unsafety {
            hir::Unsafety::Normal => {
                // OK
            }
            hir::Unsafety::Unsafe => {
                span_err!(self.tcx.sess,
                          item.span,
                          E0197,
                          "inherent impls cannot be declared as unsafe");
            }
        }

        let def_id = self.tcx.hir.local_def_id(item.id);
        let self_ty = self.tcx.item_type(def_id);
        match self_ty.sty {
            ty::TyAdt(def, _) => {
                self.check_def_id(item, def.did);
            }
            ty::TyDynamic(ref data, ..) if data.principal().is_some() => {
                self.check_def_id(item, data.principal().unwrap().def_id());
            }
            ty::TyChar => {
                self.check_primitive_impl(def_id,
                                          self.tcx.lang_items.char_impl(),
                                          "char",
                                          "char",
                                          item.span);
            }
            ty::TyStr => {
                self.check_primitive_impl(def_id,
                                          self.tcx.lang_items.str_impl(),
                                          "str",
                                          "str",
                                          item.span);
            }
            ty::TySlice(_) => {
                self.check_primitive_impl(def_id,
                                          self.tcx.lang_items.slice_impl(),
                                          "slice",
                                          "[T]",
                                          item.span);
            }
            ty::TyRawPtr(ty::TypeAndMut { ty: _, mutbl: hir::MutImmutable }) => {
                self.check_primitive_impl(def_id,
                                          self.tcx.lang_items.const_ptr_impl(),
                                          "const_ptr",
                                          "*const T",
                                          item.span);
            }
            ty::TyRawPtr(ty::TypeAndMut { ty: _, mutbl: hir::MutMutable }) => {
                self.check_primitive_impl(def_id,
                                          self.tcx.lang_items.mut_ptr_impl(),
                                          "mut_ptr",
                                          "*mut T",
                                          item.span);
            }
            ty::TyInt(ast::IntTy::I8) => {
                self.check_primitive_impl(def_id,
                                          self.tcx.lang_items.i8_impl(),
                                          "i8",
                                          "i8",
                                          item.span);
            }
            ty::TyInt(ast::IntTy::I16) => {
                self.check_primitive_impl(def_id,
                                          self.tcx.lang_items.i16_impl(),
                                          "i16",
                                          "i16",
                                          item.span);
            }
            ty::TyInt(ast::IntTy::I32) => {
                self.check_primitive_impl(def_id,
                                          self.tcx.lang_items.i32_impl(),
                                          "i32",
                                          "i32",
                                          item.span);
            }
            ty::TyInt(ast::IntTy::I64) => {
                self.check_primitive_impl(def_id,
                                          self.tcx.lang_items.i64_impl(),
                                          "i64",
                                          "i64",
                                          item.span);
            }
            ty::TyInt(ast::IntTy::I128) => {
                self.check_primitive_impl(def_id,
                                          self.tcx.lang_items.i128_impl(),
                                          "i128",
                                          "i128",
                                          item.span);
            }
            ty::TyInt(ast::IntTy::Is) => {
                self.check_primitive_impl(def_id,
                                          self.tcx.lang_items.isize_impl(),
                                          "isize",
                                          "isize",
                                          item.span);
            }
            ty::TyUint(ast::UintTy::U8) => {
                self.check_primitive_impl(def_id,
                                          self.tcx.lang_items.u8_impl(),
                                          "u8",
                                          "u8",
                                          item.span);
            }
            ty::TyUint(ast::UintTy::U16) => {
                self.check_primitive_impl(def_id,
                                          self.tcx.lang_items.u16_impl(),
                                          "u16",
                                          "u16",
                                          item.span);
            }
            ty::TyUint(ast::UintTy::U32) => {
                self.check_primitive_impl(def_id,
                                          self.tcx.lang_items.u32_impl(),
                                          "u32",
                                          "u32",
                                          item.span);
            }
            ty::TyUint(ast::UintTy::U64) => {
                self.check_primitive_impl(def_id,
                                          self.tcx.lang_items.u64_impl(),
                                          "u64",
                                          "u64",
                                          item.span);
            }
            ty::TyUint(ast::UintTy::U128) => {
                self.check_primitive_impl(def_id,
                                          self.tcx.lang_items.u128_impl(),
                                          "u128",
                                          "u128",
                                          item.span);
            }
            ty::TyUint(ast::UintTy::Us) => {
                self.check_primitive_impl(def_id,
                                          self.tcx.lang_items.usize_impl(),
                                          "usize",
                                          "usize",
                                          item.span);
            }
            ty::TyFloat(ast::FloatTy::F32) => {
                self.check_primitive_impl(def_id,
                                          self.tcx.lang_items.f32_impl(),
                                          "f32",
                                          "f32",
                                          item.span);
            }
            ty::TyFloat(ast::FloatTy::F64) => {
                self.check_primitive_impl(def_id,
                                          self.tcx.lang_items.f64_impl(),
                                          "f64",
                                          "f64",
                                          item.span);
            }
            ty::TyError => {
                return;
            }
            _ => {
                struct_span_err!(self.tcx.sess,
                                 ty.span,
                                 E0118,
                                 "no base type found for inherent implementation")
                    .span_label(ty.span, &format!("impl requires a base type"))
                    .note(&format!("either implement a trait on it or create a newtype \
                                    to wrap it instead"))
                    .emit();
                return;
            }
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
    }
}

impl<'a, 'tcx> InherentCollect<'a, 'tcx> {
    fn check_def_id(&self, item: &hir::Item, def_id: DefId) {
        if def_id.is_local() {
            // Add the implementation to the mapping from implementation to base
            // type def ID, if there is a base type for this implementation and
            // the implementation does not have any associated traits.
            let impl_def_id = self.tcx.hir.local_def_id(item.id);

            // Subtle: it'd be better to collect these into a local map
            // and then write the vector only once all items are known,
            // but that leads to degenerate dep-graphs. The problem is
            // that the write of that big vector winds up having reads
            // from *all* impls in the krate, since we've lost the
            // precision basically.  This would be ok in the firewall
            // model so once we've made progess towards that we can modify
            // the strategy here. In the meantime, using `push` is ok
            // because we are doing this as a pre-pass before anyone
            // actually reads from `inherent_impls` -- and we know this is
            // true beacuse we hold the refcell lock.
            self.tcx.maps.inherent_impls.borrow_mut().push(def_id, impl_def_id);
        } else {
            struct_span_err!(self.tcx.sess,
                             item.span,
                             E0116,
                             "cannot define inherent `impl` for a type outside of the crate \
                              where the type is defined")
                .span_label(item.span,
                            &format!("impl for type defined outside of crate."))
                .note("define and implement a trait or new type instead")
                .emit();
        }
    }

    fn check_primitive_impl(&self,
                            impl_def_id: DefId,
                            lang_def_id: Option<DefId>,
                            lang: &str,
                            ty: &str,
                            span: Span) {
        match lang_def_id {
            Some(lang_def_id) if lang_def_id == impl_def_id => {
                // OK
            }
            _ => {
                struct_span_err!(self.tcx.sess,
                                 span,
                                 E0390,
                                 "only a single inherent implementation marked with `#[lang = \
                                  \"{}\"]` is allowed for the `{}` primitive",
                                 lang,
                                 ty)
                    .span_help(span, "consider using a trait to implement these methods")
                    .emit();
            }
        }
    }
}

struct InherentOverlapChecker<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>
}

impl<'a, 'tcx> InherentOverlapChecker<'a, 'tcx> {
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
                    let node_id = self.tcx.hir.as_local_node_id(item1).unwrap();
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

        let inherent_impls = self.tcx.maps.inherent_impls.borrow();
        let impls = match inherent_impls.get(&ty_def_id) {
            Some(impls) => impls,
            None => return,
        };

        for (i, &impl1_def_id) in impls.iter().enumerate() {
            for &impl2_def_id in &impls[(i + 1)..] {
                self.tcx.infer_ctxt((), Reveal::UserFacing).enter(|infcx| {
                    if traits::overlapping_impls(&infcx, impl1_def_id, impl2_def_id).is_some() {
                        self.check_for_common_items_in_impls(impl1_def_id, impl2_def_id)
                    }
                });
            }
        }
    }
}

impl<'a, 'tcx, 'v> ItemLikeVisitor<'v> for InherentOverlapChecker<'a, 'tcx> {
    fn visit_item(&mut self, item: &'v hir::Item) {
        match item.node {
            hir::ItemEnum(..) |
            hir::ItemStruct(..) |
            hir::ItemTrait(..) |
            hir::ItemUnion(..) => {
                let type_def_id = self.tcx.hir.local_def_id(item.id);
                self.check_for_overlapping_inherent_impls(type_def_id);
            }
            _ => {}
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &hir::TraitItem) {
    }

    fn visit_impl_item(&mut self, _impl_item: &hir::ImplItem) {
    }
}

pub fn check<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    tcx.visit_all_item_likes_in_krate(DepNode::CoherenceCheckImpl,
                                      &mut InherentCollect { tcx });
    tcx.visit_all_item_likes_in_krate(DepNode::CoherenceOverlapCheckSpecial,
                                      &mut InherentOverlapChecker { tcx });
}
