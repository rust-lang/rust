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

use middle::cstore::LOCAL_CRATE;
use middle::def_id::DefId;
use middle::traits;
use middle::ty;
use syntax::ast;
use syntax::codemap::Span;
use rustc::dep_graph::DepNode;
use rustc_front::intravisit;
use rustc_front::hir;

pub fn check(tcx: &ty::ctxt) {
    let mut orphan = OrphanChecker { tcx: tcx };
    tcx.visit_all_items_in_krate(DepNode::CoherenceOrphanCheck, &mut orphan);
}

struct OrphanChecker<'cx, 'tcx:'cx> {
    tcx: &'cx ty::ctxt<'tcx>
}

impl<'cx, 'tcx> OrphanChecker<'cx, 'tcx> {
    fn check_def_id(&self, item: &hir::Item, def_id: DefId) {
        if def_id.krate != LOCAL_CRATE {
            span_err!(self.tcx.sess, item.span, E0116,
                      "cannot define inherent `impl` for a type outside of the \
                       crate where the type is defined; define and implement \
                       a trait or new type instead");
        }
    }

    fn check_primitive_impl(&self,
                            impl_def_id: DefId,
                            lang_def_id: Option<DefId>,
                            lang: &str,
                            ty: &str,
                            span: Span) {
        match lang_def_id {
            Some(lang_def_id) if lang_def_id == impl_def_id => { /* OK */ },
            _ => {
                struct_span_err!(self.tcx.sess, span, E0390,
                          "only a single inherent implementation marked with `#[lang = \"{}\"]` \
                           is allowed for the `{}` primitive", lang, ty)
                    .span_help(span, "consider using a trait to implement these methods")
                    .emit();
            }
        }
    }

    /// Checks exactly one impl for orphan rules and other such
    /// restrictions.  In this fn, it can happen that multiple errors
    /// apply to a specific impl, so just return after reporting one
    /// to prevent inundating the user with a bunch of similar error
    /// reports.
    fn check_item(&self, item: &hir::Item) {
        let def_id = self.tcx.map.local_def_id(item.id);
        match item.node {
            hir::ItemImpl(_, _, _, None, _, _) => {
                // For inherent impls, self type must be a nominal type
                // defined in this crate.
                debug!("coherence2::orphan check: inherent impl {}",
                       self.tcx.map.node_to_string(item.id));
                let self_ty = self.tcx.lookup_item_type(def_id).ty;
                match self_ty.sty {
                    ty::TyEnum(def, _) |
                    ty::TyStruct(def, _) => {
                        self.check_def_id(item, def.did);
                    }
                    ty::TyTrait(ref data) => {
                        self.check_def_id(item, data.principal_def_id());
                    }
                    ty::TyBox(..) => {
                        match self.tcx.lang_items.require_owned_box() {
                            Ok(trait_id) => self.check_def_id(item, trait_id),
                            Err(msg) => self.tcx.sess.span_fatal(item.span, &msg),
                        }
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
                    ty::TyInt(ast::TyI8) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.i8_impl(),
                                                  "i8",
                                                  "i8",
                                                  item.span);
                    }
                    ty::TyInt(ast::TyI16) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.i16_impl(),
                                                  "i16",
                                                  "i16",
                                                  item.span);
                    }
                    ty::TyInt(ast::TyI32) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.i32_impl(),
                                                  "i32",
                                                  "i32",
                                                  item.span);
                    }
                    ty::TyInt(ast::TyI64) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.i64_impl(),
                                                  "i64",
                                                  "i64",
                                                  item.span);
                    }
                    ty::TyInt(ast::TyIs) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.isize_impl(),
                                                  "isize",
                                                  "isize",
                                                  item.span);
                    }
                    ty::TyUint(ast::TyU8) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.u8_impl(),
                                                  "u8",
                                                  "u8",
                                                  item.span);
                    }
                    ty::TyUint(ast::TyU16) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.u16_impl(),
                                                  "u16",
                                                  "u16",
                                                  item.span);
                    }
                    ty::TyUint(ast::TyU32) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.u32_impl(),
                                                  "u32",
                                                  "u32",
                                                  item.span);
                    }
                    ty::TyUint(ast::TyU64) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.u64_impl(),
                                                  "u64",
                                                  "u64",
                                                  item.span);
                    }
                    ty::TyUint(ast::TyUs) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.usize_impl(),
                                                  "usize",
                                                  "usize",
                                                  item.span);
                    }
                    ty::TyFloat(ast::TyF32) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.f32_impl(),
                                                  "f32",
                                                  "f32",
                                                  item.span);
                    }
                    ty::TyFloat(ast::TyF64) => {
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
                        span_err!(self.tcx.sess, item.span, E0118,
                                  "no base type found for inherent implementation; \
                                   implement a trait or new type instead");
                        return;
                    }
                }
            }
            hir::ItemImpl(_, _, _, Some(_), _, _) => {
                // "Trait" impl
                debug!("coherence2::orphan check: trait impl {}",
                       self.tcx.map.node_to_string(item.id));
                if let ty::TyError = self.tcx.lookup_item_type(def_id).ty.sty {
                    return;
                }
                let trait_ref = self.tcx.impl_trait_ref(def_id).unwrap();
                let trait_def_id = trait_ref.def_id;
                match traits::orphan_check(self.tcx, def_id) {
                    Ok(()) => { }
                    Err(traits::OrphanCheckErr::NoLocalInputType) => {
                        span_err!(
                            self.tcx.sess, item.span, E0117,
                            "the impl does not reference any \
                             types defined in this crate; \
                             only traits defined in the current crate can be \
                             implemented for arbitrary types");
                        return;
                    }
                    Err(traits::OrphanCheckErr::UncoveredTy(param_ty)) => {
                        span_err!(self.tcx.sess, item.span, E0210,
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
                if
                    self.tcx.trait_has_default_impl(trait_def_id) &&
                    trait_def_id.krate != LOCAL_CRATE
                {
                    let self_ty = trait_ref.self_ty();
                    let opt_self_def_id = match self_ty.sty {
                        ty::TyStruct(self_def, _) | ty::TyEnum(self_def, _) =>
                            Some(self_def.did),
                        ty::TyBox(..) =>
                            self.tcx.lang_items.owned_box(),
                        _ =>
                            None
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
                                Some(format!(
                                    "cross-crate traits with a default impl, like `{}`, \
                                     can only be implemented for a struct/enum type \
                                     defined in the current crate",
                                    self.tcx.item_path_str(trait_def_id)))
                            }
                        }
                        _ => {
                            Some(format!(
                                "cross-crate traits with a default impl, like `{}`, \
                                 can only be implemented for a struct/enum type, \
                                 not `{}`",
                                self.tcx.item_path_str(trait_def_id),
                                self_ty))
                        }
                    };

                    if let Some(msg) = msg {
                        span_err!(self.tcx.sess, item.span, E0321, "{}", msg);
                        return;
                    }
                }

                // Disallow *all* explicit impls of `Sized` and `Unsize` for now.
                if Some(trait_def_id) == self.tcx.lang_items.sized_trait() {
                    span_err!(self.tcx.sess, item.span, E0322,
                              "explicit impls for the `Sized` trait are not permitted");
                    return;
                }
                if Some(trait_def_id) == self.tcx.lang_items.unsize_trait() {
                    span_err!(self.tcx.sess, item.span, E0328,
                              "explicit impls for the `Unsize` trait are not permitted");
                    return;
                }
            }
            hir::ItemDefaultImpl(..) => {
                // "Trait" impl
                debug!("coherence2::orphan check: default trait impl {}",
                       self.tcx.map.node_to_string(item.id));
                let trait_ref = self.tcx.impl_trait_ref(def_id).unwrap();
                if trait_ref.def_id.krate != LOCAL_CRATE {
                    span_err!(self.tcx.sess, item.span, E0318,
                              "cannot create default implementations for traits outside the \
                               crate they're defined in; define a new trait instead");
                    return;
                }
            }
            _ => {
                // Not an impl
            }
        }
    }
}

impl<'cx, 'tcx,'v> intravisit::Visitor<'v> for OrphanChecker<'cx, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        self.check_item(item);
    }
}
