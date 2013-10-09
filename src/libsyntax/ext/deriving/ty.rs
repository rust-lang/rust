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
use ast::{Expr,Generics,Ident};
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use codemap::{Span,respan};
use opt_vec;

/// The types of pointers
pub enum PtrTy<'self> {
    Send, // ~
    Managed(ast::Mutability), // @[mut]
    Borrowed(Option<&'self str>, ast::Mutability), // &['lifetime] [mut]
}

/// A path, e.g. `::std::option::Option::<int>` (global). Has support
/// for type parameters and a lifetime.
pub struct Path<'self> {
    path: ~[&'self str],
    lifetime: Option<&'self str>,
    params: ~[~Ty<'self>],
    global: bool
}

impl<'self> Path<'self> {
    pub fn new<'r>(path: ~[&'r str]) -> Path<'r> {
        Path::new_(path, None, ~[], true)
    }
    pub fn new_local<'r>(path: &'r str) -> Path<'r> {
        Path::new_(~[ path ], None, ~[], false)
    }
    pub fn new_<'r>(path: ~[&'r str],
                    lifetime: Option<&'r str>,
                    params: ~[~Ty<'r>],
                    global: bool)
                    -> Path<'r> {
        Path {
            path: path,
            lifetime: lifetime,
            params: params,
            global: global
        }
    }

    pub fn to_ty(&self,
                 cx: @ExtCtxt,
                 span: Span,
                 self_ty: Ident,
                 self_generics: &Generics)
                 -> ast::Ty {
        cx.ty_path(self.to_path(cx, span, self_ty, self_generics), None)
    }
    pub fn to_path(&self,
                   cx: @ExtCtxt,
                   span: Span,
                   self_ty: Ident,
                   self_generics: &Generics)
                   -> ast::Path {
        let idents = self.path.map(|s| cx.ident_of(*s) );
        let lt = mk_lifetime(cx, span, &self.lifetime);
        let tys = self.params.map(|t| t.to_ty(cx, span, self_ty, self_generics));

        cx.path_all(span, self.global, idents, lt, tys)
    }
}

/// A type. Supports pointers (except for *), Self, and literals
pub enum Ty<'self> {
    Self,
    // &/~/@ Ty
    Ptr(~Ty<'self>, PtrTy<'self>),
    // mod::mod::Type<[lifetime], [Params...]>, including a plain type
    // parameter, and things like `int`
    Literal(Path<'self>),
    // includes nil
    Tuple(~[Ty<'self>])
}

pub fn borrowed_ptrty<'r>() -> PtrTy<'r> {
    Borrowed(None, ast::MutImmutable)
}
pub fn borrowed<'r>(ty: ~Ty<'r>) -> Ty<'r> {
    Ptr(ty, borrowed_ptrty())
}

pub fn borrowed_explicit_self<'r>() -> Option<Option<PtrTy<'r>>> {
    Some(Some(borrowed_ptrty()))
}

pub fn borrowed_self<'r>() -> Ty<'r> {
    borrowed(~Self)
}

pub fn nil_ty() -> Ty<'static> {
    Tuple(~[])
}

fn mk_lifetime(cx: @ExtCtxt, span: Span, lt: &Option<&str>) -> Option<ast::Lifetime> {
    match *lt {
        Some(ref s) => Some(cx.lifetime(span, cx.ident_of(*s))),
        None => None
    }
}

impl<'self> Ty<'self> {
    pub fn to_ty(&self,
                 cx: @ExtCtxt,
                 span: Span,
                 self_ty: Ident,
                 self_generics: &Generics)
                 -> ast::Ty {
        match *self {
            Ptr(ref ty, ref ptr) => {
                let raw_ty = ty.to_ty(cx, span, self_ty, self_generics);
                match *ptr {
                    Send => {
                        cx.ty_uniq(span, raw_ty)
                    }
                    Managed(mutbl) => {
                        cx.ty_box(span, raw_ty, mutbl)
                    }
                    Borrowed(ref lt, mutbl) => {
                        let lt = mk_lifetime(cx, span, lt);
                        cx.ty_rptr(span, raw_ty, lt, mutbl)
                    }
                }
            }
            Literal(ref p) => { p.to_ty(cx, span, self_ty, self_generics) }
            Self  => {
                cx.ty_path(self.to_path(cx, span, self_ty, self_generics), None)
            }
            Tuple(ref fields) => {
                let ty = if fields.is_empty() {
                    ast::ty_nil
                } else {
                    ast::ty_tup(fields.map(|f| f.to_ty(cx, span, self_ty, self_generics)))
                };

                cx.ty(span, ty)
            }
        }
    }

    pub fn to_path(&self,
                   cx: @ExtCtxt,
                   span: Span,
                   self_ty: Ident,
                   self_generics: &Generics)
                   -> ast::Path {
        match *self {
            Self => {
                let self_params = do self_generics.ty_params.map |ty_param| {
                    cx.ty_ident(span, ty_param.ident)
                };
                let lifetime = if self_generics.lifetimes.is_empty() {
                    None
                } else {
                    Some(*self_generics.lifetimes.get(0))
                };

                cx.path_all(span, false, ~[self_ty], lifetime,
                            opt_vec::take_vec(self_params))
            }
            Literal(ref p) => {
                p.to_path(cx, span, self_ty, self_generics)
            }
            Ptr(*) => { cx.span_bug(span, "Pointer in a path in generic `deriving`") }
            Tuple(*) => { cx.span_bug(span, "Tuple in a path in generic `deriving`") }
        }
    }
}


fn mk_ty_param(cx: @ExtCtxt, span: Span, name: &str, bounds: &[Path],
               self_ident: Ident, self_generics: &Generics) -> ast::TyParam {
    let bounds = opt_vec::from(
        do bounds.map |b| {
            let path = b.to_path(cx, span, self_ident, self_generics);
            cx.typarambound(path)
        });
    cx.typaram(cx.ident_of(name), bounds)
}

fn mk_generics(lifetimes: ~[ast::Lifetime],  ty_params: ~[ast::TyParam]) -> Generics {
    Generics {
        lifetimes: opt_vec::from(lifetimes),
        ty_params: opt_vec::from(ty_params)
    }
}

/// Lifetimes and bounds on type parameters
pub struct LifetimeBounds<'self> {
    lifetimes: ~[&'self str],
    bounds: ~[(&'self str, ~[Path<'self>])]
}

impl<'self> LifetimeBounds<'self> {
    pub fn empty() -> LifetimeBounds<'static> {
        LifetimeBounds {
            lifetimes: ~[], bounds: ~[]
        }
    }
    pub fn to_generics(&self,
                       cx: @ExtCtxt,
                       span: Span,
                       self_ty: Ident,
                       self_generics: &Generics)
                       -> Generics {
        let lifetimes = do self.lifetimes.map |lt| {
            cx.lifetime(span, cx.ident_of(*lt))
        };
        let ty_params = do self.bounds.map |t| {
            match t {
                &(ref name, ref bounds) => {
                    mk_ty_param(cx, span, *name, *bounds, self_ty, self_generics)
                }
            }
        };
        mk_generics(lifetimes, ty_params)
    }
}


pub fn get_explicit_self(cx: @ExtCtxt, span: Span, self_ptr: &Option<PtrTy>)
    -> (@Expr, ast::explicit_self) {
    let self_path = cx.expr_self(span);
    match *self_ptr {
        None => {
            (self_path, respan(span, ast::sty_value))
        }
        Some(ref ptr) => {
            let self_ty = respan(
                span,
                match *ptr {
                    Send => ast::sty_uniq,
                    Managed(mutbl) => ast::sty_box(mutbl),
                    Borrowed(ref lt, mutbl) => {
                        let lt = lt.map(|s| cx.lifetime(span, cx.ident_of(s)));
                        ast::sty_region(lt, mutbl)
                    }
                });
            let self_expr = cx.expr_deref(span, self_path);
            (self_expr, self_ty)
        }
    }
}
