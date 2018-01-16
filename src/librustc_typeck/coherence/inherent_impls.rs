// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The code in this module gathers up all of the inherent impls in
//! the current crate and organizes them in a map. It winds up
//! touching the whole crate and thus must be recomputed completely
//! for any change, but it is very cheap to compute. In practice, most
//! code in the compiler never *directly* requests this map. Instead,
//! it requests the inherent impls specific to some type (via
//! `tcx.inherent_impls(def_id)`). That value, however,
//! is computed by selecting an idea from this table.

use rustc::dep_graph::DepKind;
use rustc::hir::def_id::{CrateNum, DefId, LOCAL_CRATE};
use rustc::hir;
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::ty::{self, CrateInherentImpls, TyCtxt};
use rustc::util::nodemap::DefIdMap;

use std::rc::Rc;
use syntax::ast;
use syntax_pos::Span;

/// On-demand query: yields a map containing all types mapped to their inherent impls.
pub fn crate_inherent_impls<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                      crate_num: CrateNum)
                                      -> CrateInherentImpls {
    assert_eq!(crate_num, LOCAL_CRATE);

    let krate = tcx.hir.krate();
    let mut collect = InherentCollect {
        tcx,
        impls_map: CrateInherentImpls {
            inherent_impls: DefIdMap()
        }
    };
    krate.visit_all_item_likes(&mut collect);
    collect.impls_map
}

/// On-demand query: yields a vector of the inherent impls for a specific type.
pub fn inherent_impls<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                ty_def_id: DefId)
                                -> Rc<Vec<DefId>> {
    assert!(ty_def_id.is_local());

    // NB. Until we adopt the red-green dep-tracking algorithm (see
    // [the plan] for details on that), we do some hackery here to get
    // the dependencies correct.  Basically, we use a `with_ignore` to
    // read the result we want. If we didn't have the `with_ignore`,
    // we would wind up with a dependency on the entire crate, which
    // we don't want. Then we go and add dependencies on all the impls
    // in the result (which is what we wanted).
    //
    // The result is a graph with an edge from `Hir(I)` for every impl
    // `I` defined on some type `T` to `CoherentInherentImpls(T)`,
    // thus ensuring that if any of those impls change, the set of
    // inherent impls is considered dirty.
    //
    // [the plan]: https://github.com/rust-lang/rust-roadmap/issues/4

    thread_local! {
        static EMPTY_DEF_ID_VEC: Rc<Vec<DefId>> = Rc::new(vec![])
    }

    let result = tcx.dep_graph.with_ignore(|| {
        let crate_map = tcx.crate_inherent_impls(ty_def_id.krate);
        match crate_map.inherent_impls.get(&ty_def_id) {
            Some(v) => v.clone(),
            None => EMPTY_DEF_ID_VEC.with(|v| v.clone())
        }
    });

    for &impl_def_id in &result[..] {
        let def_path_hash = tcx.def_path_hash(impl_def_id);
        tcx.dep_graph.read(def_path_hash.to_dep_node(DepKind::Hir));
    }

    result
}

struct InherentCollect<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    impls_map: CrateInherentImpls,
}

impl<'a, 'tcx, 'v> ItemLikeVisitor<'v> for InherentCollect<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        let ty = match item.node {
            hir::ItemImpl(.., None, ref ty, _) => ty,
            _ => return
        };

        let def_id = self.tcx.hir.local_def_id(item.id);
        let self_ty = self.tcx.type_of(def_id);
        let lang_items = self.tcx.lang_items();
        match self_ty.sty {
            ty::TyAdt(def, _) => {
                self.check_def_id(item, def.did);
            }
            ty::TyForeign(did) => {
                self.check_def_id(item, did);
            }
            ty::TyDynamic(ref data, ..) if data.principal().is_some() => {
                self.check_def_id(item, data.principal().unwrap().def_id());
            }
            ty::TyChar => {
                self.check_primitive_impl(def_id,
                                          lang_items.char_impl(),
                                          "char",
                                          "char",
                                          item.span);
            }
            ty::TyStr => {
                self.check_primitive_impl(def_id,
                                          lang_items.str_impl(),
                                          "str",
                                          "str",
                                          item.span);
            }
            ty::TySlice(slice_item) if slice_item == self.tcx.types.u8 => {
                self.check_primitive_impl(def_id,
                                          lang_items.slice_u8_impl(),
                                          "slice_u8",
                                          "[u8]",
                                          item.span);
            }
            ty::TySlice(_) => {
                self.check_primitive_impl(def_id,
                                          lang_items.slice_impl(),
                                          "slice",
                                          "[T]",
                                          item.span);
            }
            ty::TyRawPtr(ty::TypeAndMut { ty: _, mutbl: hir::MutImmutable }) => {
                self.check_primitive_impl(def_id,
                                          lang_items.const_ptr_impl(),
                                          "const_ptr",
                                          "*const T",
                                          item.span);
            }
            ty::TyRawPtr(ty::TypeAndMut { ty: _, mutbl: hir::MutMutable }) => {
                self.check_primitive_impl(def_id,
                                          lang_items.mut_ptr_impl(),
                                          "mut_ptr",
                                          "*mut T",
                                          item.span);
            }
            ty::TyInt(ast::IntTy::I8) => {
                self.check_primitive_impl(def_id,
                                          lang_items.i8_impl(),
                                          "i8",
                                          "i8",
                                          item.span);
            }
            ty::TyInt(ast::IntTy::I16) => {
                self.check_primitive_impl(def_id,
                                          lang_items.i16_impl(),
                                          "i16",
                                          "i16",
                                          item.span);
            }
            ty::TyInt(ast::IntTy::I32) => {
                self.check_primitive_impl(def_id,
                                          lang_items.i32_impl(),
                                          "i32",
                                          "i32",
                                          item.span);
            }
            ty::TyInt(ast::IntTy::I64) => {
                self.check_primitive_impl(def_id,
                                          lang_items.i64_impl(),
                                          "i64",
                                          "i64",
                                          item.span);
            }
            ty::TyInt(ast::IntTy::I128) => {
                self.check_primitive_impl(def_id,
                                          lang_items.i128_impl(),
                                          "i128",
                                          "i128",
                                          item.span);
            }
            ty::TyInt(ast::IntTy::Isize) => {
                self.check_primitive_impl(def_id,
                                          lang_items.isize_impl(),
                                          "isize",
                                          "isize",
                                          item.span);
            }
            ty::TyUint(ast::UintTy::U8) => {
                self.check_primitive_impl(def_id,
                                          lang_items.u8_impl(),
                                          "u8",
                                          "u8",
                                          item.span);
            }
            ty::TyUint(ast::UintTy::U16) => {
                self.check_primitive_impl(def_id,
                                          lang_items.u16_impl(),
                                          "u16",
                                          "u16",
                                          item.span);
            }
            ty::TyUint(ast::UintTy::U32) => {
                self.check_primitive_impl(def_id,
                                          lang_items.u32_impl(),
                                          "u32",
                                          "u32",
                                          item.span);
            }
            ty::TyUint(ast::UintTy::U64) => {
                self.check_primitive_impl(def_id,
                                          lang_items.u64_impl(),
                                          "u64",
                                          "u64",
                                          item.span);
            }
            ty::TyUint(ast::UintTy::U128) => {
                self.check_primitive_impl(def_id,
                                          lang_items.u128_impl(),
                                          "u128",
                                          "u128",
                                          item.span);
            }
            ty::TyUint(ast::UintTy::Usize) => {
                self.check_primitive_impl(def_id,
                                          lang_items.usize_impl(),
                                          "usize",
                                          "usize",
                                          item.span);
            }
            ty::TyFloat(ast::FloatTy::F32) => {
                self.check_primitive_impl(def_id,
                                          lang_items.f32_impl(),
                                          "f32",
                                          "f32",
                                          item.span);
            }
            ty::TyFloat(ast::FloatTy::F64) => {
                self.check_primitive_impl(def_id,
                                          lang_items.f64_impl(),
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
                    .span_label(ty.span, "impl requires a base type")
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
    fn check_def_id(&mut self, item: &hir::Item, def_id: DefId) {
        if def_id.is_local() {
            // Add the implementation to the mapping from implementation to base
            // type def ID, if there is a base type for this implementation and
            // the implementation does not have any associated traits.
            let impl_def_id = self.tcx.hir.local_def_id(item.id);
            let mut rc_vec = self.impls_map.inherent_impls
                                           .entry(def_id)
                                           .or_insert_with(|| Rc::new(vec![]));

            // At this point, there should not be any clones of the
            // `Rc`, so we can still safely push into it in place:
            Rc::get_mut(&mut rc_vec).unwrap().push(impl_def_id);
        } else {
            struct_span_err!(self.tcx.sess,
                             item.span,
                             E0116,
                             "cannot define inherent `impl` for a type outside of the crate \
                              where the type is defined")
                .span_label(item.span,
                            "impl for type defined outside of crate.")
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
