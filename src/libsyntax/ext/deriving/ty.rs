// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
A mini version of ast::Ty, which is easier to use, and features an
explicit `Self` type to use when specifying impls to be derived.
*/

use ast;
use ast::{expr,Generics,ident};
use ext::base::ext_ctxt;
use ext::build;
use codemap::{span,respan};
use opt_vec;

/// The types of pointers
#[deriving(Eq)]
pub enum PtrTy {
    Owned, // ~
    Managed(ast::mutability), // @[mut]
    Borrowed(Option<~str>, ast::mutability), // &['lifetime] [mut]
}

/// A path, e.g. `::core::option::Option::<int>` (global). Has support
/// for type parameters and a lifetime.
#[deriving(Eq)]
pub struct Path {
    path: ~[~str],
    lifetime: Option<~str>,
    params: ~[~Ty],
    global: bool
}

pub impl Path {
    fn new(path: ~[~str]) -> Path {
        Path::new_(path, None, ~[], true)
    }
    fn new_local(path: ~str) -> Path {
        Path::new_(~[ path ], None, ~[], false)
    }
    fn new_(path: ~[~str], lifetime: Option<~str>, params: ~[~Ty], global: bool) -> Path {
        Path {
            path: path,
            lifetime: lifetime,
            params: params,
            global: global
        }
    }

    fn to_ty(&self, cx: @ext_ctxt, span: span,
             self_ty: ident, self_generics: &Generics) -> @ast::Ty {
                build::mk_ty_path_path(cx, span,
                                       self.to_path(cx, span,
                                                    self_ty, self_generics))
    }
    fn to_path(&self, cx: @ext_ctxt, span: span,
               self_ty: ident, self_generics: &Generics) -> @ast::Path {
        let idents = self.path.map(|s| cx.ident_of(*s) );
        let lt = mk_lifetime(cx, span, &self.lifetime);
        let tys = self.params.map(|t| t.to_ty(cx, span, self_ty, self_generics));

        if self.global {
            build::mk_raw_path_global_(span, idents, lt, tys)
        } else {
            build::mk_raw_path_(span, idents, lt, tys)
        }
    }
}

/// A type. Supports pointers (except for *), Self, and literals
#[deriving(Eq)]
pub enum Ty {
    Self,
    // &/~/@ Ty
    Ptr(~Ty, PtrTy),
    // mod::mod::Type<[lifetime], [Params...]>, including a plain type
    // parameter, and things like `int`
    Literal(Path),
    // includes nil
    Tuple(~[Ty])
}

pub fn borrowed_ptrty() -> PtrTy {
    Borrowed(None, ast::m_imm)
}
pub fn borrowed(ty: ~Ty) -> Ty {
    Ptr(ty, borrowed_ptrty())
}

pub fn borrowed_explicit_self() -> Option<Option<PtrTy>> {
    Some(Some(borrowed_ptrty()))
}

pub fn borrowed_self() -> Ty {
    borrowed(~Self)
}

pub fn nil_ty() -> Ty {
    Tuple(~[])
}

fn mk_lifetime(cx: @ext_ctxt, span: span, lt: &Option<~str>) -> Option<@ast::Lifetime> {
    match *lt {
        Some(ref s) => Some(@build::mk_lifetime(cx, span, cx.ident_of(*s))),
        None => None
    }
}

pub impl Ty {
    fn to_ty(&self, cx: @ext_ctxt, span: span,
             self_ty: ident, self_generics: &Generics) -> @ast::Ty {
        match *self {
            Ptr(ref ty, ref ptr) => {
                let raw_ty = ty.to_ty(cx, span, self_ty, self_generics);
                match *ptr {
                    Owned => {
                        build::mk_ty_uniq(cx, span, raw_ty)
                    }
                    Managed(mutbl) => {
                        build::mk_ty_box(cx, span, raw_ty, mutbl)
                    }
                    Borrowed(ref lt, mutbl) => {
                        let lt = mk_lifetime(cx, span, lt);
                        build::mk_ty_rptr(cx, span, raw_ty, lt, mutbl)
                    }
                }
            }
            Literal(ref p) => { p.to_ty(cx, span, self_ty, self_generics) }
            Self  => {
                build::mk_ty_path_path(cx, span, self.to_path(cx, span, self_ty, self_generics))
            }
            Tuple(ref fields) => {
                let ty = if fields.is_empty() {
                    ast::ty_nil
                } else {
                    ast::ty_tup(fields.map(|f| f.to_ty(cx, span, self_ty, self_generics)))
                };

                build::mk_ty(cx, span, ty)
            }
        }
    }

    fn to_path(&self, cx: @ext_ctxt, span: span,
               self_ty: ident, self_generics: &Generics) -> @ast::Path {
        match *self {
            Self => {
                let self_params = do self_generics.ty_params.map |ty_param| {
                    build::mk_ty_path(cx, span, ~[ ty_param.ident ])
                };
                let lifetime = if self_generics.lifetimes.is_empty() {
                    None
                } else {
                    Some(@*self_generics.lifetimes.get(0))
                };

                build::mk_raw_path_(span, ~[self_ty], lifetime,
                                    opt_vec::take_vec(self_params))
            }
            Literal(ref p) => {
                p.to_path(cx, span, self_ty, self_generics)
            }
            Ptr(*) => { cx.span_bug(span, ~"Pointer in a path in generic `deriving`") }
            Tuple(*) => { cx.span_bug(span, ~"Tuple in a path in generic `deriving`") }
        }
    }
}


fn mk_ty_param(cx: @ext_ctxt, span: span, name: ~str, bounds: ~[Path],
               self_ident: ident, self_generics: &Generics) -> ast::TyParam {
    let bounds = opt_vec::from(
        do bounds.map |b| {
            let path = b.to_path(cx, span, self_ident, self_generics);
            build::mk_trait_ty_param_bound_(cx, path)
        });
    build::mk_ty_param(cx, cx.ident_of(name), @bounds)
}

fn mk_generics(lifetimes: ~[ast::Lifetime],  ty_params: ~[ast::TyParam]) -> Generics {
    Generics {
        lifetimes: opt_vec::from(lifetimes),
        ty_params: opt_vec::from(ty_params)
    }
}

/// Lifetimes and bounds on type parameters
pub struct LifetimeBounds {
    lifetimes: ~[~str],
    bounds: ~[(~str, ~[Path])]
}

pub impl LifetimeBounds {
    fn empty() -> LifetimeBounds {
        LifetimeBounds {
            lifetimes: ~[], bounds: ~[]
        }
    }
    fn to_generics(&self, cx: @ext_ctxt, span: span,
                   self_ty: ident, self_generics: &Generics) -> Generics {
        let lifetimes = do self.lifetimes.map |&lt| {
            build::mk_lifetime(cx, span, cx.ident_of(lt))
        };
        let ty_params = do self.bounds.map |&(name, bounds)| {
            mk_ty_param(cx, span, name, bounds, self_ty, self_generics)
        };
        mk_generics(lifetimes, ty_params)
    }
}


pub fn get_explicit_self(cx: @ext_ctxt, span: span, self_ptr: &Option<PtrTy>)
    -> (@expr, ast::self_ty) {
    let self_path = build::make_self(cx, span);
    match *self_ptr {
        None => {
            (self_path, respan(span, ast::sty_value))
        }
        Some(ref ptr) => {
            let self_ty = respan(
                span,
                match *ptr {
                    Owned => ast::sty_uniq(ast::m_imm),
                    Managed(mutbl) => ast::sty_box(mutbl),
                    Borrowed(ref lt, mutbl) => {
                        let lt = lt.map(|s| @build::mk_lifetime(cx, span,
                                                                cx.ident_of(*s)));
                        ast::sty_region(lt, mutbl)
                    }
                });
            let self_expr = build::mk_deref(cx, span, self_path);
            (self_expr, self_ty)
        }
    }
}
