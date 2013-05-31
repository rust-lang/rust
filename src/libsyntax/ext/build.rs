// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use abi::AbiSet;
use ast::ident;
use ast;
use ast_util;
use codemap::{span, respan, dummy_sp};
use fold;
use ext::base::ExtCtxt;
use ext::quote::rt::*;
use opt_vec;
use opt_vec::OptVec;

pub struct Field {
    ident: ast::ident,
    ex: @ast::expr
}

// Transitional reexports so qquote can find the paths it is looking for
mod syntax {
    pub use ext;
    pub use parse;
}

pub trait AstBuilder {
    // paths
    fn path(&self, span: span, strs: ~[ast::ident]) -> @ast::Path;
    fn path_ident(&self, span: span, id: ast::ident) -> @ast::Path;
    fn path_global(&self, span: span, strs: ~[ast::ident]) -> @ast::Path;
    fn path_all(&self, sp: span,
                global: bool,
                idents: ~[ast::ident],
                rp: Option<@ast::Lifetime>,
                types: ~[@ast::Ty])
        -> @ast::Path;

    // types
    fn ty_mt(&self, ty: @ast::Ty, mutbl: ast::mutability) -> ast::mt;

    fn ty(&self, span: span, ty: ast::ty_) -> @ast::Ty;
    fn ty_path(&self, @ast::Path) -> @ast::Ty;
    fn ty_ident(&self, span: span, idents: ast::ident) -> @ast::Ty;

    fn ty_rptr(&self, span: span,
               ty: @ast::Ty,
               lifetime: Option<@ast::Lifetime>,
               mutbl: ast::mutability)
        -> @ast::Ty;
    fn ty_uniq(&self, span: span, ty: @ast::Ty) -> @ast::Ty;
    fn ty_box(&self, span: span, ty: @ast::Ty, mutbl: ast::mutability) -> @ast::Ty;

    fn ty_option(&self, ty: @ast::Ty) -> @ast::Ty;
    fn ty_infer(&self, sp: span) -> @ast::Ty;
    fn ty_nil(&self) -> @ast::Ty;

    fn ty_vars(&self, ty_params: &OptVec<ast::TyParam>) -> ~[@ast::Ty];
    fn ty_vars_global(&self, ty_params: &OptVec<ast::TyParam>) -> ~[@ast::Ty];
    fn ty_field_imm(&self, span: span, name: ident, ty: @ast::Ty) -> ast::ty_field;
    fn strip_bounds(&self, bounds: &Generics) -> Generics;

    fn typaram(&self, id: ast::ident, bounds: @OptVec<ast::TyParamBound>) -> ast::TyParam;

    fn trait_ref(&self, path: @ast::Path) -> @ast::trait_ref;
    fn typarambound(&self, path: @ast::Path) -> ast::TyParamBound;
    fn lifetime(&self, span: span, ident: ast::ident) -> ast::Lifetime;

    // statements
    fn stmt_expr(&self, expr: @ast::expr) -> @ast::stmt;
    fn stmt_let(&self, sp: span, mutbl: bool, ident: ast::ident, ex: @ast::expr) -> @ast::stmt;

    // blocks
    fn blk(&self, span: span, stmts: ~[@ast::stmt], expr: Option<@ast::expr>) -> ast::blk;
    fn blk_expr(&self, expr: @ast::expr) -> ast::blk;
    fn blk_all(&self, span: span,
               view_items: ~[@ast::view_item],
               stmts: ~[@ast::stmt],
               expr: Option<@ast::expr>) -> ast::blk;

    // expressions
    fn expr(&self, span: span, node: ast::expr_) -> @ast::expr;
    fn expr_path(&self, path: @ast::Path) -> @ast::expr;
    fn expr_ident(&self, span: span, id: ast::ident) -> @ast::expr;

    fn expr_self(&self, span: span) -> @ast::expr;
    fn expr_binary(&self, sp: span, op: ast::binop,
                   lhs: @ast::expr, rhs: @ast::expr) -> @ast::expr;
    fn expr_deref(&self, sp: span, e: @ast::expr) -> @ast::expr;
    fn expr_unary(&self, sp: span, op: ast::unop, e: @ast::expr) -> @ast::expr;

    fn expr_copy(&self, sp: span, e: @ast::expr) -> @ast::expr;
    fn expr_managed(&self, sp: span, e: @ast::expr) -> @ast::expr;
    fn expr_addr_of(&self, sp: span, e: @ast::expr) -> @ast::expr;
    fn expr_mut_addr_of(&self, sp: span, e: @ast::expr) -> @ast::expr;
    fn expr_field_access(&self, span: span, expr: @ast::expr, ident: ast::ident) -> @ast::expr;
    fn expr_call(&self, span: span, expr: @ast::expr, args: ~[@ast::expr]) -> @ast::expr;
    fn expr_call_ident(&self, span: span, id: ast::ident, args: ~[@ast::expr]) -> @ast::expr;
    fn expr_call_global(&self, sp: span, fn_path: ~[ast::ident],
                        args: ~[@ast::expr]) -> @ast::expr;
    fn expr_method_call(&self, span: span,
                        expr: @ast::expr, ident: ast::ident,
                        args: ~[@ast::expr]) -> @ast::expr;
    fn expr_blk(&self, b: ast::blk) -> @ast::expr;

    fn field_imm(&self, span: span, name: ident, e: @ast::expr) -> ast::field;
    fn expr_struct(&self, span: span, path: @ast::Path, fields: ~[ast::field]) -> @ast::expr;
    fn expr_struct_ident(&self, span: span, id: ast::ident, fields: ~[ast::field]) -> @ast::expr;

    fn expr_lit(&self, sp: span, lit: ast::lit_) -> @ast::expr;

    fn expr_uint(&self, span: span, i: uint) -> @ast::expr;
    fn expr_int(&self, sp: span, i: int) -> @ast::expr;
    fn expr_u8(&self, sp: span, u: u8) -> @ast::expr;
    fn expr_bool(&self, sp: span, value: bool) -> @ast::expr;

    fn expr_vstore(&self, sp: span, expr: @ast::expr, vst: ast::expr_vstore) -> @ast::expr;
    fn expr_vec(&self, sp: span, exprs: ~[@ast::expr]) -> @ast::expr;
    fn expr_vec_uniq(&self, sp: span, exprs: ~[@ast::expr]) -> @ast::expr;
    fn expr_vec_slice(&self, sp: span, exprs: ~[@ast::expr]) -> @ast::expr;
    fn expr_str(&self, sp: span, s: ~str) -> @ast::expr;
    fn expr_str_uniq(&self, sp: span, s: ~str) -> @ast::expr;

    fn expr_unreachable(&self, span: span) -> @ast::expr;

    fn pat(&self, span: span, pat: ast::pat_) -> @ast::pat;
    fn pat_wild(&self, span: span) -> @ast::pat;
    fn pat_lit(&self, span: span, expr: @ast::expr) -> @ast::pat;
    fn pat_ident(&self, span: span, ident: ast::ident) -> @ast::pat;

    fn pat_ident_binding_mode(&self,
                              span: span,
                              ident: ast::ident,
                              bm: ast::binding_mode) -> @ast::pat;
    fn pat_enum(&self, span: span, path: @ast::Path, subpats: ~[@ast::pat]) -> @ast::pat;
    fn pat_struct(&self, span: span,
                  path: @ast::Path, field_pats: ~[ast::field_pat]) -> @ast::pat;

    fn arm(&self, span: span, pats: ~[@ast::pat], expr: @ast::expr) -> ast::arm;
    fn arm_unreachable(&self, span: span) -> ast::arm;

    fn expr_match(&self, span: span, arg: @ast::expr, arms: ~[ast::arm]) -> @ast::expr;
    fn expr_if(&self, span: span,
               cond: @ast::expr, then: @ast::expr, els: Option<@ast::expr>) -> @ast::expr;

    fn lambda_fn_decl(&self, span: span, fn_decl: ast::fn_decl, blk: ast::blk) -> @ast::expr;

    fn lambda(&self, span: span, ids: ~[ast::ident], blk: ast::blk) -> @ast::expr;
    fn lambda0(&self, span: span, blk: ast::blk) -> @ast::expr;
    fn lambda1(&self, span: span, blk: ast::blk, ident: ast::ident) -> @ast::expr;

    fn lambda_expr(&self, span: span, ids: ~[ast::ident], blk: @ast::expr) -> @ast::expr;
    fn lambda_expr_0(&self, span: span, expr: @ast::expr) -> @ast::expr;
    fn lambda_expr_1(&self, span: span, expr: @ast::expr, ident: ast::ident) -> @ast::expr;

    fn lambda_stmts(&self, span: span, ids: ~[ast::ident], blk: ~[@ast::stmt]) -> @ast::expr;
    fn lambda_stmts_0(&self, span: span, stmts: ~[@ast::stmt]) -> @ast::expr;
    fn lambda_stmts_1(&self, span: span, stmts: ~[@ast::stmt], ident: ast::ident) -> @ast::expr;

    // items
    fn item(&self, span: span,
            name: ident, attrs: ~[ast::attribute], node: ast::item_) -> @ast::item;

    fn arg(&self, span: span, name: ident, ty: @ast::Ty) -> ast::arg;
    // XXX unused self
    fn fn_decl(&self, inputs: ~[ast::arg], output: @ast::Ty) -> ast::fn_decl;

    fn item_fn_poly(&self,
                    span: span,
                    name: ident,
                    inputs: ~[ast::arg],
                    output: @ast::Ty,
                    generics: Generics,
                    body: ast::blk) -> @ast::item;
    fn item_fn(&self,
               span: span,
               name: ident,
               inputs: ~[ast::arg],
               output: @ast::Ty,
               body: ast::blk) -> @ast::item;

    fn variant(&self, span: span, name: ident, tys: ~[@ast::Ty]) -> ast::variant;
    fn item_enum_poly(&self,
                      span: span,
                      name: ident,
                      enum_definition: ast::enum_def,
                      generics: Generics) -> @ast::item;
    fn item_enum(&self, span: span, name: ident, enum_def: ast::enum_def) -> @ast::item;

    fn item_struct_poly(&self,
                        span: span,
                        name: ident,
                        struct_def: ast::struct_def,
                        generics: Generics) -> @ast::item;
    fn item_struct(&self, span: span, name: ident, struct_def: ast::struct_def) -> @ast::item;

    fn item_mod(&self, span: span,
                name: ident, attrs: ~[ast::attribute],
                vi: ~[@ast::view_item], items: ~[@ast::item]) -> @ast::item;

    fn item_ty_poly(&self,
                    span: span,
                    name: ident,
                    ty: @ast::Ty,
                    generics: Generics) -> @ast::item;
    fn item_ty(&self, span: span, name: ident, ty: @ast::Ty) -> @ast::item;

    fn attribute(&self, sp: span, mi: @ast::meta_item) -> ast::attribute;

    fn meta_word(&self, sp: span, w: ~str) -> @ast::meta_item;
    fn meta_list(&self, sp: span, name: ~str, mis: ~[@ast::meta_item]) -> @ast::meta_item;
    fn meta_name_value(&self, sp: span, name: ~str, value: ast::lit_) -> @ast::meta_item;

    fn view_use(&self, sp: span,
                vis: ast::visibility, vp: ~[@ast::view_path]) -> @ast::view_item;
    fn view_use_list(&self, sp: span, vis: ast::visibility,
                     path: ~[ast::ident], imports: &[ast::ident]) -> @ast::view_item;
    fn view_use_glob(&self, sp: span,
                     vis: ast::visibility, path: ~[ast::ident]) -> @ast::view_item;
}

impl AstBuilder for @ExtCtxt {
    fn path(&self, span: span, strs: ~[ast::ident]) -> @ast::Path {
        self.path_all(span, false, strs, None, ~[])
    }
    fn path_ident(&self, span: span, id: ast::ident) -> @ast::Path {
        self.path(span, ~[id])
    }
    fn path_global(&self, span: span, strs: ~[ast::ident]) -> @ast::Path {
        self.path_all(span, true, strs, None, ~[])
    }
    fn path_all(&self, sp: span,
                global: bool,
                idents: ~[ast::ident],
                rp: Option<@ast::Lifetime>,
                types: ~[@ast::Ty])
        -> @ast::Path {
        @ast::Path {
            span: sp,
            global: global,
            idents: idents,
            rp: rp,
            types: types
        }
    }

    fn ty_mt(&self, ty: @ast::Ty, mutbl: ast::mutability) -> ast::mt {
        ast::mt {
            ty: ty,
            mutbl: mutbl
        }
    }

    fn ty(&self, span: span, ty: ast::ty_) -> @ast::Ty {
        @ast::Ty {
            id: self.next_id(),
            span: span,
            node: ty
        }
    }

    fn ty_path(&self, path: @ast::Path) -> @ast::Ty {
        self.ty(path.span,
                ast::ty_path(path, self.next_id()))
    }

    fn ty_ident(&self, span: span, ident: ast::ident)
        -> @ast::Ty {
        self.ty_path(self.path_ident(span, ident))
    }

    fn ty_rptr(&self,
               span: span,
               ty: @ast::Ty,
               lifetime: Option<@ast::Lifetime>,
               mutbl: ast::mutability)
        -> @ast::Ty {
        self.ty(span,
                ast::ty_rptr(lifetime, self.ty_mt(ty, mutbl)))
    }
    fn ty_uniq(&self, span: span, ty: @ast::Ty) -> @ast::Ty {
        self.ty(span, ast::ty_uniq(self.ty_mt(ty, ast::m_imm)))
    }
    fn ty_box(&self, span: span,
                 ty: @ast::Ty, mutbl: ast::mutability) -> @ast::Ty {
        self.ty(span, ast::ty_box(self.ty_mt(ty, mutbl)))
    }

    fn ty_option(&self, ty: @ast::Ty) -> @ast::Ty {
        self.ty_path(
            self.path_all(dummy_sp(),
                          true,
                          ~[
                              self.ident_of("std"),
                              self.ident_of("option"),
                              self.ident_of("Option")
                          ],
                          None,
                          ~[ ty ]))
    }

    fn ty_field_imm(&self, span: span, name: ident, ty: @ast::Ty) -> ast::ty_field {
        respan(span,
               ast::ty_field_ {
                   ident: name,
                   mt: ast::mt { ty: ty, mutbl: ast::m_imm },
               })
    }

    fn ty_infer(&self, span: span) -> @ast::Ty {
        self.ty(span, ast::ty_infer)
    }

    fn ty_nil(&self) -> @ast::Ty {
        @ast::Ty {
            id: self.next_id(),
            node: ast::ty_nil,
            span: dummy_sp(),
        }
    }

    fn typaram(&self, id: ast::ident, bounds: @OptVec<ast::TyParamBound>) -> ast::TyParam {
        ast::TyParam { ident: id, id: self.next_id(), bounds: bounds }
    }

    // these are strange, and probably shouldn't be used outside of
    // pipes. Specifically, the global version possible generates
    // incorrect code.
    fn ty_vars(&self, ty_params: &OptVec<ast::TyParam>) -> ~[@ast::Ty] {
        opt_vec::take_vec(
            ty_params.map(|p| self.ty_ident(dummy_sp(), p.ident)))
    }

    fn ty_vars_global(&self, ty_params: &OptVec<ast::TyParam>) -> ~[@ast::Ty] {
        opt_vec::take_vec(
            ty_params.map(|p| self.ty_path(
                self.path_global(dummy_sp(), ~[p.ident]))))
    }

    fn strip_bounds(&self, generics: &Generics) -> Generics {
        let no_bounds = @opt_vec::Empty;
        let new_params = do generics.ty_params.map |ty_param| {
            ast::TyParam { bounds: no_bounds, ..copy *ty_param }
        };
        Generics {
            ty_params: new_params,
            .. copy *generics
        }
    }

    fn trait_ref(&self, path: @ast::Path) -> @ast::trait_ref {
        @ast::trait_ref {
            path: path,
            ref_id: self.next_id()
        }
    }

    fn typarambound(&self, path: @ast::Path) -> ast::TyParamBound {
        ast::TraitTyParamBound(self.trait_ref(path))
    }

    fn lifetime(&self, span: span, ident: ast::ident) -> ast::Lifetime {
        ast::Lifetime { id: self.next_id(), span: span, ident: ident }
    }

    fn stmt_expr(&self, expr: @ast::expr) -> @ast::stmt {
        @respan(expr.span, ast::stmt_semi(expr, self.next_id()))
    }

    fn stmt_let(&self, sp: span, mutbl: bool, ident: ast::ident, ex: @ast::expr) -> @ast::stmt {
        let pat = self.pat_ident(sp, ident);
        let local = @respan(sp,
                            ast::local_ {
                                is_mutbl: mutbl,
                                ty: self.ty_infer(sp),
                                pat: pat,
                                init: Some(ex),
                                id: self.next_id(),
                            });
        let decl = respan(sp, ast::decl_local(~[local]));
        @respan(sp, ast::stmt_decl(@decl, self.next_id()))
    }

    fn blk(&self, span: span, stmts: ~[@ast::stmt], expr: Option<@expr>) -> ast::blk {
        self.blk_all(span, ~[], stmts, expr)
    }

    fn blk_expr(&self, expr: @ast::expr) -> ast::blk {
        self.blk_all(expr.span, ~[], ~[], Some(expr))
    }
    fn blk_all(&self,
               span: span,
               view_items: ~[@ast::view_item],
               stmts: ~[@ast::stmt],
               expr: Option<@ast::expr>) -> ast::blk {
        respan(span,
               ast::blk_ {
                   view_items: view_items,
                   stmts: stmts,
                   expr: expr,
                   id: self.next_id(),
                   rules: ast::default_blk,
               })
    }

    fn expr(&self, span: span, node: ast::expr_) -> @ast::expr {
        @ast::expr {
            id: self.next_id(),
            callee_id: self.next_id(),
            node: node,
            span: span,
        }
    }

    fn expr_path(&self, path: @ast::Path) -> @ast::expr {
        self.expr(path.span, ast::expr_path(path))
    }

    fn expr_ident(&self, span: span, id: ast::ident) -> @ast::expr {
        self.expr_path(self.path_ident(span, id))
    }
    fn expr_self(&self, span: span) -> @ast::expr {
        self.expr(span, ast::expr_self)
    }

    fn expr_binary(&self, sp: span, op: ast::binop,
                   lhs: @ast::expr, rhs: @ast::expr) -> @ast::expr {
        self.next_id(); // see ast_util::op_expr_callee_id
        self.expr(sp, ast::expr_binary(op, lhs, rhs))
    }

    fn expr_deref(&self, sp: span, e: @ast::expr) -> @ast::expr {
        self.expr_unary(sp, ast::deref, e)
    }
    fn expr_unary(&self, sp: span, op: ast::unop, e: @ast::expr)
        -> @ast::expr {
        self.next_id(); // see ast_util::op_expr_callee_id
        self.expr(sp, ast::expr_unary(op, e))
    }

    fn expr_copy(&self, sp: span, e: @ast::expr) -> @ast::expr {
        self.expr(sp, ast::expr_copy(e))
    }
    fn expr_managed(&self, sp: span, e: @ast::expr) -> @ast::expr {
        self.expr_unary(sp, ast::box(ast::m_imm), e)
    }

    fn expr_field_access(&self, sp: span, expr: @ast::expr, ident: ast::ident) -> @ast::expr {
        self.expr(sp, ast::expr_field(expr, ident, ~[]))
    }
    fn expr_addr_of(&self, sp: span, e: @ast::expr) -> @ast::expr {
        self.expr(sp, ast::expr_addr_of(ast::m_imm, e))
    }
    fn expr_mut_addr_of(&self, sp: span, e: @ast::expr) -> @ast::expr {
        self.expr(sp, ast::expr_addr_of(ast::m_mutbl, e))
    }

    fn expr_call(&self, span: span, expr: @ast::expr, args: ~[@ast::expr]) -> @ast::expr {
        self.expr(span, ast::expr_call(expr, args, ast::NoSugar))
    }
    fn expr_call_ident(&self, span: span, id: ast::ident, args: ~[@ast::expr]) -> @ast::expr {
        self.expr(span,
                  ast::expr_call(self.expr_ident(span, id), args, ast::NoSugar))
    }
    fn expr_call_global(&self, sp: span, fn_path: ~[ast::ident],
                      args: ~[@ast::expr]) -> @ast::expr {
        let pathexpr = self.expr_path(self.path_global(sp, fn_path));
        self.expr_call(sp, pathexpr, args)
    }
    fn expr_method_call(&self, span: span,
                        expr: @ast::expr,
                        ident: ast::ident,
                        args: ~[@ast::expr]) -> @ast::expr {
        self.expr(span,
                  ast::expr_method_call(expr, ident, ~[], args, ast::NoSugar))
    }
    fn expr_blk(&self, b: ast::blk) -> @ast::expr {
        self.expr(b.span, ast::expr_block(b))
    }
    fn field_imm(&self, span: span, name: ident, e: @ast::expr) -> ast::field {
        respan(span, ast::field_ { ident: name, expr: e })
    }
    fn expr_struct(&self, span: span, path: @ast::Path, fields: ~[ast::field]) -> @ast::expr {
        self.expr(span, ast::expr_struct(path, fields, None))
    }
    fn expr_struct_ident(&self, span: span,
                         id: ast::ident, fields: ~[ast::field]) -> @ast::expr {
        self.expr_struct(span, self.path_ident(span, id), fields)
    }

    fn expr_lit(&self, sp: span, lit: ast::lit_) -> @ast::expr {
        self.expr(sp, ast::expr_lit(@respan(sp, lit)))
    }
    fn expr_uint(&self, span: span, i: uint) -> @ast::expr {
        self.expr_lit(span, ast::lit_uint(i as u64, ast::ty_u))
    }
    fn expr_int(&self, sp: span, i: int) -> @ast::expr {
        self.expr_lit(sp, ast::lit_int(i as i64, ast::ty_i))
    }
    fn expr_u8(&self, sp: span, u: u8) -> @ast::expr {
        self.expr_lit(sp, ast::lit_uint(u as u64, ast::ty_u8))
    }
    fn expr_bool(&self, sp: span, value: bool) -> @ast::expr {
        self.expr_lit(sp, ast::lit_bool(value))
    }

    fn expr_vstore(&self, sp: span, expr: @ast::expr, vst: ast::expr_vstore) -> @ast::expr {
        self.expr(sp, ast::expr_vstore(expr, vst))
    }
    fn expr_vec(&self, sp: span, exprs: ~[@ast::expr]) -> @ast::expr {
        self.expr(sp, ast::expr_vec(exprs, ast::m_imm))
    }
    fn expr_vec_uniq(&self, sp: span, exprs: ~[@ast::expr]) -> @ast::expr {
        self.expr_vstore(sp, self.expr_vec(sp, exprs), ast::expr_vstore_uniq)
    }
    fn expr_vec_slice(&self, sp: span, exprs: ~[@ast::expr]) -> @ast::expr {
        self.expr_vstore(sp, self.expr_vec(sp, exprs), ast::expr_vstore_slice)
    }
    fn expr_str(&self, sp: span, s: ~str) -> @ast::expr {
        self.expr_lit(sp, ast::lit_str(@s))
    }
    fn expr_str_uniq(&self, sp: span, s: ~str) -> @ast::expr {
        self.expr_vstore(sp, self.expr_str(sp, s), ast::expr_vstore_uniq)
    }


    fn expr_unreachable(&self, span: span) -> @ast::expr {
        let loc = self.codemap().lookup_char_pos(span.lo);
        self.expr_call_global(
            span,
            ~[
                self.ident_of("std"),
                self.ident_of("sys"),
                self.ident_of("FailWithCause"),
                self.ident_of("fail_with"),
            ],
            ~[
                self.expr_str(span, ~"internal error: entered unreachable code"),
                self.expr_str(span, copy loc.file.name),
                self.expr_uint(span, loc.line),
            ])
    }


    fn pat(&self, span: span, pat: ast::pat_) -> @ast::pat {
        @ast::pat { id: self.next_id(), node: pat, span: span }
    }
    fn pat_wild(&self, span: span) -> @ast::pat {
        self.pat(span, ast::pat_wild)
    }
    fn pat_lit(&self, span: span, expr: @ast::expr) -> @ast::pat {
        self.pat(span, ast::pat_lit(expr))
    }
    fn pat_ident(&self, span: span, ident: ast::ident) -> @ast::pat {
        self.pat_ident_binding_mode(span, ident, ast::bind_infer)
    }

    fn pat_ident_binding_mode(&self,
                              span: span,
                              ident: ast::ident,
                              bm: ast::binding_mode) -> @ast::pat {
        let path = self.path_ident(span, ident);
        let pat = ast::pat_ident(bm, path, None);
        self.pat(span, pat)
    }
    fn pat_enum(&self, span: span, path: @ast::Path, subpats: ~[@ast::pat]) -> @ast::pat {
        let pat = ast::pat_enum(path, Some(subpats));
        self.pat(span, pat)
    }
    fn pat_struct(&self, span: span,
                  path: @ast::Path, field_pats: ~[ast::field_pat]) -> @ast::pat {
        let pat = ast::pat_struct(path, field_pats, false);
        self.pat(span, pat)
    }

    fn arm(&self, _span: span, pats: ~[@ast::pat], expr: @ast::expr) -> ast::arm {
        ast::arm {
            pats: pats,
            guard: None,
            body: self.blk_expr(expr)
        }
    }

    fn arm_unreachable(&self, span: span) -> ast::arm {
        self.arm(span, ~[self.pat_wild(span)], self.expr_unreachable(span))
    }

    fn expr_match(&self, span: span, arg: @ast::expr, arms: ~[ast::arm]) -> @expr {
        self.expr(span, ast::expr_match(arg, arms))
    }

    fn expr_if(&self, span: span,
               cond: @ast::expr, then: @ast::expr, els: Option<@ast::expr>) -> @ast::expr {
        let els = els.map(|x| self.expr_blk(self.blk_expr(*x)));
        self.expr(span, ast::expr_if(cond, self.blk_expr(then), els))
    }

    fn lambda_fn_decl(&self, span: span, fn_decl: ast::fn_decl, blk: ast::blk) -> @ast::expr {
        self.expr(span, ast::expr_fn_block(fn_decl, blk))
    }
    fn lambda(&self, span: span, ids: ~[ast::ident], blk: ast::blk) -> @ast::expr {
        let fn_decl = self.fn_decl(
            ids.map(|id| self.arg(span, *id, self.ty_infer(span))),
            self.ty_infer(span));

        self.expr(span, ast::expr_fn_block(fn_decl, blk))
    }
    fn lambda0(&self, _span: span, blk: ast::blk) -> @ast::expr {
        let ext_cx = *self;
        let blk_e = self.expr(copy blk.span, ast::expr_block(copy blk));
        quote_expr!(|| $blk_e )
    }

    fn lambda1(&self, _span: span, blk: ast::blk, ident: ast::ident) -> @ast::expr {
        let ext_cx = *self;
        let blk_e = self.expr(copy blk.span, ast::expr_block(copy blk));
        quote_expr!(|$ident| $blk_e )
    }

    fn lambda_expr(&self, span: span, ids: ~[ast::ident], expr: @ast::expr) -> @ast::expr {
        self.lambda(span, ids, self.blk_expr(expr))
    }
    fn lambda_expr_0(&self, span: span, expr: @ast::expr) -> @ast::expr {
        self.lambda0(span, self.blk_expr(expr))
    }
    fn lambda_expr_1(&self, span: span, expr: @ast::expr, ident: ast::ident) -> @ast::expr {
        self.lambda1(span, self.blk_expr(expr), ident)
    }

    fn lambda_stmts(&self, span: span, ids: ~[ast::ident], stmts: ~[@ast::stmt]) -> @ast::expr {
        self.lambda(span, ids, self.blk(span, stmts, None))
    }
    fn lambda_stmts_0(&self, span: span, stmts: ~[@ast::stmt]) -> @ast::expr {
        self.lambda0(span, self.blk(span, stmts, None))
    }
    fn lambda_stmts_1(&self, span: span, stmts: ~[@ast::stmt], ident: ast::ident) -> @ast::expr {
        self.lambda1(span, self.blk(span, stmts, None), ident)
    }

    fn arg(&self, span: span, ident: ast::ident, ty: @ast::Ty) -> ast::arg {
        let arg_pat = self.pat_ident(span, ident);
        ast::arg {
            is_mutbl: false,
            ty: ty,
            pat: arg_pat,
            id: self.next_id()
        }
    }

    // XXX unused self
    fn fn_decl(&self, inputs: ~[ast::arg], output: @ast::Ty) -> ast::fn_decl {
        ast::fn_decl {
            inputs: inputs,
            output: output,
            cf: ast::return_val,
        }
    }

    fn item(&self, span: span,
            name: ident, attrs: ~[ast::attribute], node: ast::item_) -> @ast::item {
        // XXX: Would be nice if our generated code didn't violate
        // Rust coding conventions
        @ast::item { ident: name,
                    attrs: attrs,
                    id: self.next_id(),
                    node: node,
                    vis: ast::public,
                    span: span }
    }

    fn item_fn_poly(&self,
                    span: span,
                    name: ident,
                    inputs: ~[ast::arg],
                    output: @ast::Ty,
                    generics: Generics,
                    body: ast::blk) -> @ast::item {
        self.item(span,
                  name,
                  ~[],
                  ast::item_fn(self.fn_decl(inputs, output),
                               ast::impure_fn,
                               AbiSet::Rust(),
                               generics,
                               body))
    }

    fn item_fn(&self,
               span: span,
               name: ident,
               inputs: ~[ast::arg],
               output: @ast::Ty,
               body: ast::blk
              ) -> @ast::item {
        self.item_fn_poly(
            span,
            name,
            inputs,
            output,
            ast_util::empty_generics(),
            body)
    }

    fn variant(&self, span: span, name: ident, tys: ~[@ast::Ty]) -> ast::variant {
        let args = do tys.map |ty| {
            ast::variant_arg { ty: *ty, id: self.next_id() }
        };

        respan(span,
               ast::variant_ {
                   name: name,
                   attrs: ~[],
                   kind: ast::tuple_variant_kind(args),
                   id: self.next_id(),
                   disr_expr: None,
                   vis: ast::public
               })
    }

    fn item_enum_poly(&self, span: span, name: ident,
                      enum_definition: ast::enum_def,
                      generics: Generics) -> @ast::item {
        self.item(span, name, ~[], ast::item_enum(enum_definition, generics))
    }

    fn item_enum(&self, span: span, name: ident,
                 enum_definition: ast::enum_def) -> @ast::item {
        self.item_enum_poly(span, name, enum_definition,
                            ast_util::empty_generics())
    }

    fn item_struct(
        &self,
        span: span,
        name: ident,
        struct_def: ast::struct_def
    ) -> @ast::item {
        self.item_struct_poly(
            span,
            name,
            struct_def,
            ast_util::empty_generics()
        )
    }

    fn item_struct_poly(
        &self,
        span: span,
        name: ident,
        struct_def: ast::struct_def,
        generics: Generics
    ) -> @ast::item {
        self.item(span, name, ~[], ast::item_struct(@struct_def, generics))
    }

    fn item_mod(&self, span: span, name: ident,
                attrs: ~[ast::attribute],
                vi: ~[@ast::view_item],
                items: ~[@ast::item]) -> @ast::item {
        self.item(
            span,
            name,
            attrs,
            ast::item_mod(ast::_mod {
                view_items: vi,
                items: items,
            })
        )
    }

    fn item_ty_poly(&self, span: span, name: ident, ty: @ast::Ty,
                    generics: Generics) -> @ast::item {
        self.item(span, name, ~[], ast::item_ty(ty, generics))
    }

    fn item_ty(&self, span: span, name: ident, ty: @ast::Ty) -> @ast::item {
        self.item_ty_poly(span, name, ty, ast_util::empty_generics())
    }

    fn attribute(&self, sp: span, mi: @ast::meta_item) -> ast::attribute {
        respan(sp,
               ast::attribute_ {
                   style: ast::attr_outer,
                   value: mi,
                   is_sugared_doc: false
               })
    }

    fn meta_word(&self, sp: span, w: ~str) -> @ast::meta_item {
        @respan(sp, ast::meta_word(@w))
    }
    fn meta_list(&self, sp: span, name: ~str, mis: ~[@ast::meta_item]) -> @ast::meta_item {
        @respan(sp, ast::meta_list(@name, mis))
    }
    fn meta_name_value(&self, sp: span, name: ~str, value: ast::lit_) -> @ast::meta_item {
        @respan(sp, ast::meta_name_value(@name, respan(sp, value)))
    }

    fn view_use(&self, sp: span,
                vis: ast::visibility, vp: ~[@ast::view_path]) -> @ast::view_item {
        @ast::view_item {
            node: ast::view_item_use(vp),
            attrs: ~[],
            vis: vis,
            span: sp
        }
    }

    fn view_use_list(&self, sp: span, vis: ast::visibility,
                     path: ~[ast::ident], imports: &[ast::ident]) -> @ast::view_item {
        let imports = do imports.map |id| {
            respan(sp, ast::path_list_ident_ { name: *id, id: self.next_id() })
        };

        self.view_use(sp, vis,
                      ~[@respan(sp,
                                ast::view_path_list(self.path(sp, path),
                                                    imports,
                                                    self.next_id()))])
    }

    fn view_use_glob(&self, sp: span,
                     vis: ast::visibility, path: ~[ast::ident]) -> @ast::view_item {
        self.view_use(sp, vis,
                      ~[@respan(sp,
                                ast::view_path_glob(self.path(sp, path), self.next_id()))])
    }
}


pub trait Duplicate {
    //
    // Duplication functions
    //
    // These functions just duplicate AST nodes.
    //

    fn duplicate(&self, cx: @ExtCtxt) -> Self;
}

impl Duplicate for @ast::expr {
    fn duplicate(&self, cx: @ExtCtxt) -> @ast::expr {
        let folder = fold::default_ast_fold();
        let folder = @fold::AstFoldFns {
            new_id: |_| cx.next_id(),
            ..*folder
        };
        let folder = fold::make_fold(folder);
        folder.fold_expr(*self)
    }
}
