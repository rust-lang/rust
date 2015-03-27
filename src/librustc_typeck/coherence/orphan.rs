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
    fn check_def_id(&self, item: &ast::Item, def_id: ast::DefId) {
        if def_id.krate != ast::LOCAL_CRATE {
            span_err!(self.tcx.sess, item.span, E0116,
                      "cannot associate methods with a type outside the \
                       crate the type is defined in; define and implement \
                       a trait or new type instead");
        }
    }

    fn check_primitive_impl(&self,
                            impl_def_id: ast::DefId,
                            lang_def_id: Option<ast::DefId>,
                            lang: &str,
                            ty: &str,
                            span: Span) {
        match lang_def_id {
            Some(lang_def_id) if lang_def_id == impl_def_id => { /* OK */ },
            _ => {
                self.tcx.sess.span_err(
                    span,
                    &format!("only a single inherent implementation marked with `#[lang = \"{}\"]` \
                              is allowed for the `{}` primitive", lang, ty));
            }
        }
    }

    /// Checks exactly one impl for orphan rules and other such
    /// restrictions.  In this fn, it can happen that multiple errors
    /// apply to a specific impl, so just return after reporting one
    /// to prevent inundating the user with a bunch of similar error
    /// reports.
    fn check_item(&self, item: &ast::Item) {
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
                        self.check_def_id(item, def_id);
                    }
                    ty::ty_trait(ref data) => {
                        self.check_def_id(item, data.principal_def_id());
                    }
                    ty::ty_uniq(..) => {
                        self.check_def_id(item, self.tcx.lang_items.owned_box().unwrap());
                    }
                    ty::ty_char => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.char_impl(),
                                                  "char",
                                                  "char",
                                                  item.span);
                    }
                    ty::ty_str => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.str_impl(),
                                                  "str",
                                                  "str",
                                                  item.span);
                    }
                    ty::ty_vec(_, None) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.slice_impl(),
                                                  "slice",
                                                  "[T]",
                                                  item.span);
                    }
                    ty::ty_ptr(ty::mt { ty: _, mutbl: ast::MutImmutable }) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.const_ptr_impl(),
                                                  "const_ptr",
                                                  "*const T",
                                                  item.span);
                    }
                    ty::ty_ptr(ty::mt { ty: _, mutbl: ast::MutMutable }) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.mut_ptr_impl(),
                                                  "mut_ptr",
                                                  "*mut T",
                                                  item.span);
                    }
                    ty::ty_int(ast::TyI8) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.i8_impl(),
                                                  "i8",
                                                  "i8",
                                                  item.span);
                    }
                    ty::ty_int(ast::TyI16) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.i16_impl(),
                                                  "i16",
                                                  "i16",
                                                  item.span);
                    }
                    ty::ty_int(ast::TyI32) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.i32_impl(),
                                                  "i32",
                                                  "i32",
                                                  item.span);
                    }
                    ty::ty_int(ast::TyI64) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.i64_impl(),
                                                  "i64",
                                                  "i64",
                                                  item.span);
                    }
                    ty::ty_int(ast::TyIs(_)) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.isize_impl(),
                                                  "isize",
                                                  "isize",
                                                  item.span);
                    }
                    ty::ty_uint(ast::TyU8) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.u8_impl(),
                                                  "u8",
                                                  "u8",
                                                  item.span);
                    }
                    ty::ty_uint(ast::TyU16) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.u16_impl(),
                                                  "u16",
                                                  "u16",
                                                  item.span);
                    }
                    ty::ty_uint(ast::TyU32) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.u32_impl(),
                                                  "u32",
                                                  "u32",
                                                  item.span);
                    }
                    ty::ty_uint(ast::TyU64) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.u64_impl(),
                                                  "u64",
                                                  "u64",
                                                  item.span);
                    }
                    ty::ty_uint(ast::TyUs(_)) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.usize_impl(),
                                                  "usize",
                                                  "usize",
                                                  item.span);
                    }
                    ty::ty_float(ast::TyF32) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.f32_impl(),
                                                  "f32",
                                                  "f32",
                                                  item.span);
                    }
                    ty::ty_float(ast::TyF64) => {
                        self.check_primitive_impl(def_id,
                                                  self.tcx.lang_items.f64_impl(),
                                                  "f64",
                                                  "f64",
                                                  item.span);
                    }
                    _ => {
                        span_err!(self.tcx.sess, item.span, E0118,
                                  "no base type found for inherent implementation; \
                                   implement a trait or new type instead");
                        return;
                    }
                }
            }
            ast::ItemImpl(_, _, _, Some(_), _, _) => {
                // "Trait" impl
                debug!("coherence2::orphan check: trait impl {}", item.repr(self.tcx));
                let trait_ref = ty::impl_trait_ref(self.tcx, def_id).unwrap();
                let trait_def_id = trait_ref.def_id;
                match traits::orphan_check(self.tcx, def_id) {
                    Ok(()) => { }
                    Err(traits::OrphanCheckErr::NoLocalInputType) => {
                        if !ty::has_attr(self.tcx, trait_def_id, "old_orphan_check") {
                            span_err!(
                                self.tcx.sess, item.span, E0117,
                                "the impl does not reference any \
                                 types defined in this crate; \
                                 only traits defined in the current crate can be \
                                 implemented for arbitrary types");
                            return;
                        }
                    }
                    Err(traits::OrphanCheckErr::UncoveredTy(param_ty)) => {
                        if !ty::has_attr(self.tcx, trait_def_id, "old_orphan_check") {
                            span_err!(self.tcx.sess, item.span, E0210,
                                    "type parameter `{}` must be used as the type parameter for \
                                     some local type (e.g. `MyStruct<T>`); only traits defined in \
                                     the current crate can be implemented for a type parameter",
                                    param_ty.user_string(self.tcx));
                            return;
                        }
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
                debug!("trait_ref={} trait_def_id={} trait_has_default_impl={}",
                       trait_ref.repr(self.tcx),
                       trait_def_id.repr(self.tcx),
                       ty::trait_has_default_impl(self.tcx, trait_def_id));
                if
                    ty::trait_has_default_impl(self.tcx, trait_def_id) &&
                    trait_def_id.krate != ast::LOCAL_CRATE
                {
                    let self_ty = trait_ref.self_ty();
                    let opt_self_def_id = match self_ty.sty {
                        ty::ty_struct(self_def_id, _) | ty::ty_enum(self_def_id, _) =>
                            Some(self_def_id),
                        ty::ty_uniq(..) =>
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
                            if self_def_id.krate == ast::LOCAL_CRATE {
                                None
                            } else {
                                Some(format!(
                                    "cross-crate traits with a default impl, like `{}`, \
                                     can only be implemented for a struct/enum type \
                                     defined in the current crate",
                                    ty::item_path_str(self.tcx, trait_def_id)))
                            }
                        }
                        _ => {
                            Some(format!(
                                "cross-crate traits with a default impl, like `{}`, \
                                 can only be implemented for a struct/enum type, \
                                 not `{}`",
                                ty::item_path_str(self.tcx, trait_def_id),
                                self_ty.user_string(self.tcx)))
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
                    span_err!(self.tcx.sess, item.span, E0323,
                              "explicit impls for the `Unsize` trait are not permitted");
                    return;
                }
            }
            ast::ItemDefaultImpl(..) => {
                // "Trait" impl
                debug!("coherence2::orphan check: default trait impl {}", item.repr(self.tcx));
                let trait_ref = ty::impl_trait_ref(self.tcx, def_id).unwrap();
                if trait_ref.def_id.krate != ast::LOCAL_CRATE {
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

impl<'cx, 'tcx,'v> visit::Visitor<'v> for OrphanChecker<'cx, 'tcx> {
    fn visit_item(&mut self, item: &ast::Item) {
        self.check_item(item);
        visit::walk_item(self, item);
    }
}
