// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::AbiSet;
use ast::ident;
use ast;
use ast_util;
use codemap;
use codemap::{span, respan, dummy_sp, spanned};
use fold;
use ext::base::ExtCtxt;
use ext::quote::rt::*;
use opt_vec;
use opt_vec::OptVec;

pub struct Field {
    ident: ast::ident,
    ex: @ast::expr
}

pub fn mk_expr(cx: @ExtCtxt,
               sp: codemap::span,
               expr: ast::expr_)
            -> @ast::expr {
    @ast::expr {
        id: cx.next_id(),
        callee_id: cx.next_id(),
        node: expr,
        span: sp,
    }
}

pub fn mk_lit(cx: @ExtCtxt, sp: span, lit: ast::lit_) -> @ast::expr {
    let sp_lit = @codemap::spanned { node: lit, span: sp };
    mk_expr(cx, sp, ast::expr_lit(sp_lit))
}
pub fn mk_int(cx: @ExtCtxt, sp: span, i: int) -> @ast::expr {
    let lit = ast::lit_int(i as i64, ast::ty_i);
    return mk_lit(cx, sp, lit);
}
pub fn mk_uint(cx: @ExtCtxt, sp: span, u: uint) -> @ast::expr {
    let lit = ast::lit_uint(u as u64, ast::ty_u);
    return mk_lit(cx, sp, lit);
}
pub fn mk_u8(cx: @ExtCtxt, sp: span, u: u8) -> @ast::expr {
    let lit = ast::lit_uint(u as u64, ast::ty_u8);
    return mk_lit(cx, sp, lit);
}
pub fn mk_binary(cx: @ExtCtxt, sp: span, op: ast::binop,
                 lhs: @ast::expr, rhs: @ast::expr) -> @ast::expr {
    cx.next_id(); // see ast_util::op_expr_callee_id
    mk_expr(cx, sp, ast::expr_binary(op, lhs, rhs))
}

pub fn mk_deref(cx: @ExtCtxt, sp: span, e: @ast::expr) -> @ast::expr {
    mk_unary(cx, sp, ast::deref, e)
}
pub fn mk_unary(cx: @ExtCtxt, sp: span, op: ast::unop, e: @ast::expr)
             -> @ast::expr {
    cx.next_id(); // see ast_util::op_expr_callee_id
    mk_expr(cx, sp, ast::expr_unary(op, e))
}
pub fn mk_raw_path(sp: span, idents: ~[ast::ident]) -> @ast::Path {
    mk_raw_path_(sp, idents, None, ~[])
}
pub fn mk_raw_path_(sp: span,
                    idents: ~[ast::ident],
                    rp: Option<@ast::Lifetime>,
                    types: ~[@ast::Ty])
                 -> @ast::Path {
    @ast::Path { span: sp,
                 global: false,
                 idents: idents,
                 rp: rp,
                 types: types }
}
pub fn mk_raw_path_global(sp: span, idents: ~[ast::ident]) -> @ast::Path {
    mk_raw_path_global_(sp, idents, None, ~[])
}
pub fn mk_raw_path_global_(sp: span,
                           idents: ~[ast::ident],
                           rp: Option<@ast::Lifetime>,
                           types: ~[@ast::Ty]) -> @ast::Path {
    @ast::Path { span: sp,
                 global: true,
                 idents: idents,
                 rp: rp,
                 types: types }
}
pub fn mk_path_raw(cx: @ExtCtxt, sp: span, path: @ast::Path)-> @ast::expr {
    mk_expr(cx, sp, ast::expr_path(path))
}
pub fn mk_path(cx: @ExtCtxt, sp: span, idents: ~[ast::ident])
            -> @ast::expr {
    mk_path_raw(cx, sp, mk_raw_path(sp, idents))
}
pub fn mk_path_global(cx: @ExtCtxt, sp: span, idents: ~[ast::ident])
                   -> @ast::expr {
    mk_path_raw(cx, sp, mk_raw_path_global(sp, idents))
}
pub fn mk_access_(cx: @ExtCtxt, sp: span, p: @ast::expr, m: ast::ident)
               -> @ast::expr {
    mk_expr(cx, sp, ast::expr_field(p, m, ~[]))
}
pub fn mk_access(cx: @ExtCtxt, sp: span, p: ~[ast::ident], m: ast::ident)
              -> @ast::expr {
    let pathexpr = mk_path(cx, sp, p);
    return mk_access_(cx, sp, pathexpr, m);
}
pub fn mk_addr_of(cx: @ExtCtxt, sp: span, e: @ast::expr) -> @ast::expr {
    return mk_expr(cx, sp, ast::expr_addr_of(ast::m_imm, e));
}
pub fn mk_mut_addr_of(cx: @ExtCtxt, sp: span, e: @ast::expr) -> @ast::expr {
    return mk_expr(cx, sp, ast::expr_addr_of(ast::m_mutbl, e));
}
pub fn mk_method_call(cx: @ExtCtxt,
                      sp: span,
                      rcvr_expr: @ast::expr,
                      method_ident: ast::ident,
                      args: ~[@ast::expr]) -> @ast::expr {
    mk_expr(cx, sp, ast::expr_method_call(rcvr_expr, method_ident, ~[], args, ast::NoSugar))
}
pub fn mk_call_(cx: @ExtCtxt, sp: span, fn_expr: @ast::expr,
                args: ~[@ast::expr]) -> @ast::expr {
    mk_expr(cx, sp, ast::expr_call(fn_expr, args, ast::NoSugar))
}
pub fn mk_call(cx: @ExtCtxt, sp: span, fn_path: ~[ast::ident],
               args: ~[@ast::expr]) -> @ast::expr {
    let pathexpr = mk_path(cx, sp, fn_path);
    return mk_call_(cx, sp, pathexpr, args);
}
pub fn mk_call_global(cx: @ExtCtxt, sp: span, fn_path: ~[ast::ident],
                      args: ~[@ast::expr]) -> @ast::expr {
    let pathexpr = mk_path_global(cx, sp, fn_path);
    return mk_call_(cx, sp, pathexpr, args);
}
// e = expr, t = type
pub fn mk_base_vec_e(cx: @ExtCtxt, sp: span, exprs: ~[@ast::expr])
                  -> @ast::expr {
    let vecexpr = ast::expr_vec(exprs, ast::m_imm);
    mk_expr(cx, sp, vecexpr)
}
pub fn mk_vstore_e(cx: @ExtCtxt, sp: span, expr: @ast::expr,
                   vst: ast::expr_vstore) ->
   @ast::expr {
    mk_expr(cx, sp, ast::expr_vstore(expr, vst))
}
pub fn mk_uniq_vec_e(cx: @ExtCtxt, sp: span, exprs: ~[@ast::expr])
                  -> @ast::expr {
    mk_vstore_e(cx, sp, mk_base_vec_e(cx, sp, exprs), ast::expr_vstore_uniq)
}
pub fn mk_slice_vec_e(cx: @ExtCtxt, sp: span, exprs: ~[@ast::expr])
                   -> @ast::expr {
    mk_vstore_e(cx, sp, mk_base_vec_e(cx, sp, exprs),
                ast::expr_vstore_slice)
}
pub fn mk_base_str(cx: @ExtCtxt, sp: span, s: ~str) -> @ast::expr {
    let lit = ast::lit_str(@s);
    return mk_lit(cx, sp, lit);
}
pub fn mk_uniq_str(cx: @ExtCtxt, sp: span, s: ~str) -> @ast::expr {
    mk_vstore_e(cx, sp, mk_base_str(cx, sp, s), ast::expr_vstore_uniq)
}
pub fn mk_field(sp: span, f: &Field) -> ast::field {
    codemap::spanned {
        node: ast::field_ { mutbl: ast::m_imm, ident: f.ident, expr: f.ex },
        span: sp,
    }
}
pub fn mk_fields(sp: span, fields: ~[Field]) -> ~[ast::field] {
    fields.map(|f| mk_field(sp, f))
}
pub fn mk_struct_e(cx: @ExtCtxt,
                   sp: span,
                   ctor_path: ~[ast::ident],
                   fields: ~[Field])
                -> @ast::expr {
    mk_expr(cx, sp,
            ast::expr_struct(mk_raw_path(sp, ctor_path),
                             mk_fields(sp, fields),
                                    option::None::<@ast::expr>))
}
pub fn mk_global_struct_e(cx: @ExtCtxt,
                          sp: span,
                          ctor_path: ~[ast::ident],
                          fields: ~[Field])
                       -> @ast::expr {
    mk_expr(cx, sp,
            ast::expr_struct(mk_raw_path_global(sp, ctor_path),
                             mk_fields(sp, fields),
                                    option::None::<@ast::expr>))
}
pub fn mk_glob_use(cx: @ExtCtxt,
                   sp: span,
                   vis: ast::visibility,
                   path: ~[ast::ident]) -> @ast::view_item {
    let glob = @codemap::spanned {
        node: ast::view_path_glob(mk_raw_path(sp, path), cx.next_id()),
        span: sp,
    };
    @ast::view_item { node: ast::view_item_use(~[glob]),
                      attrs: ~[],
                      vis: vis,
                      span: sp }
}
pub fn mk_local(cx: @ExtCtxt, sp: span, mutbl: bool,
                ident: ast::ident, ex: @ast::expr) -> @ast::stmt {

    let pat = @ast::pat {
        id: cx.next_id(),
        node: ast::pat_ident(
            ast::bind_by_copy,
            mk_raw_path(sp, ~[ident]),
            None),
        span: sp,
    };
    let ty = @ast::Ty { id: cx.next_id(), node: ast::ty_infer, span: sp };
    let local = @codemap::spanned {
        node: ast::local_ {
            is_mutbl: mutbl,
            ty: ty,
            pat: pat,
            init: Some(ex),
            id: cx.next_id(),
        },
        span: sp,
    };
    let decl = codemap::spanned {node: ast::decl_local(~[local]), span: sp};
    @codemap::spanned { node: ast::stmt_decl(@decl, cx.next_id()), span: sp }
}
pub fn mk_block(cx: @ExtCtxt, span: span,
                view_items: ~[@ast::view_item],
                stmts: ~[@ast::stmt],
                expr: Option<@ast::expr>) -> @ast::expr {
    let blk = codemap::spanned {
        node: ast::blk_ {
             view_items: view_items,
             stmts: stmts,
             expr: expr,
             id: cx.next_id(),
             rules: ast::default_blk,
        },
        span: span,
    };
    mk_expr(cx, span, ast::expr_block(blk))
}
pub fn mk_block_(cx: @ExtCtxt,
                 span: span,
                 stmts: ~[@ast::stmt])
              -> ast::blk {
    codemap::spanned {
        node: ast::blk_ {
            view_items: ~[],
            stmts: stmts,
            expr: None,
            id: cx.next_id(),
            rules: ast::default_blk,
        },
        span: span,
    }
}
pub fn mk_simple_block(cx: @ExtCtxt,
                       span: span,
                       expr: @ast::expr)
                    -> ast::blk {
    codemap::spanned {
        node: ast::blk_ {
            view_items: ~[],
            stmts: ~[],
            expr: Some(expr),
            id: cx.next_id(),
            rules: ast::default_blk,
        },
        span: span,
    }
}
pub fn mk_lambda_(cx: @ExtCtxt,
                 span: span,
                 fn_decl: ast::fn_decl,
                 blk: ast::blk)
              -> @ast::expr {
    mk_expr(cx, span, ast::expr_fn_block(fn_decl, blk))
}
pub fn mk_lambda(cx: @ExtCtxt,
                 span: span,
                 fn_decl: ast::fn_decl,
                 expr: @ast::expr)
              -> @ast::expr {
    let blk = mk_simple_block(cx, span, expr);
    mk_lambda_(cx, span, fn_decl, blk)
}
pub fn mk_lambda_stmts(cx: @ExtCtxt,
                       span: span,
                       fn_decl: ast::fn_decl,
                       stmts: ~[@ast::stmt])
                    -> @ast::expr {
    let blk = mk_block(cx, span, ~[], stmts, None);
    mk_lambda(cx, span, fn_decl, blk)
}
pub fn mk_lambda_no_args(cx: @ExtCtxt,
                         span: span,
                         expr: @ast::expr)
                      -> @ast::expr {
    let fn_decl = mk_fn_decl(~[], mk_ty_infer(cx, span));
    mk_lambda(cx, span, fn_decl, expr)
}
pub fn mk_copy(cx: @ExtCtxt, sp: span, e: @ast::expr) -> @ast::expr {
    mk_expr(cx, sp, ast::expr_copy(e))
}
pub fn mk_managed(cx: @ExtCtxt, sp: span, e: @ast::expr) -> @ast::expr {
    mk_expr(cx, sp, ast::expr_unary(ast::box(ast::m_imm), e))
}
pub fn mk_pat(cx: @ExtCtxt, span: span, pat: ast::pat_) -> @ast::pat {
    @ast::pat { id: cx.next_id(), node: pat, span: span }
}
pub fn mk_pat_wild(cx: @ExtCtxt, span: span) -> @ast::pat {
    mk_pat(cx, span, ast::pat_wild)
}
pub fn mk_pat_lit(cx: @ExtCtxt,
                  span: span,
                  expr: @ast::expr) -> @ast::pat {
    mk_pat(cx, span, ast::pat_lit(expr))
}
pub fn mk_pat_ident(cx: @ExtCtxt,
                    span: span,
                    ident: ast::ident) -> @ast::pat {
    mk_pat_ident_with_binding_mode(cx, span, ident, ast::bind_by_copy)
}

pub fn mk_pat_ident_with_binding_mode(cx: @ExtCtxt,
                                      span: span,
                                      ident: ast::ident,
                                      bm: ast::binding_mode) -> @ast::pat {
    let path = mk_raw_path(span, ~[ ident ]);
    let pat = ast::pat_ident(bm, path, None);
    mk_pat(cx, span, pat)
}
pub fn mk_pat_enum(cx: @ExtCtxt,
                   span: span,
                   path: @ast::Path,
                   subpats: ~[@ast::pat])
                -> @ast::pat {
    let pat = ast::pat_enum(path, Some(subpats));
    mk_pat(cx, span, pat)
}
pub fn mk_pat_struct(cx: @ExtCtxt,
                     span: span,
                     path: @ast::Path,
                     field_pats: ~[ast::field_pat])
                  -> @ast::pat {
    let pat = ast::pat_struct(path, field_pats, false);
    mk_pat(cx, span, pat)
}
pub fn mk_bool(cx: @ExtCtxt, span: span, value: bool) -> @ast::expr {
    let lit_expr = ast::expr_lit(@codemap::spanned {
        node: ast::lit_bool(value),
        span: span });
    build::mk_expr(cx, span, lit_expr)
}
pub fn mk_stmt(cx: @ExtCtxt, span: span, expr: @ast::expr) -> @ast::stmt {
    let stmt_ = ast::stmt_semi(expr, cx.next_id());
    @codemap::spanned { node: stmt_, span: span }
}

pub fn mk_ty_mt(ty: @ast::Ty, mutbl: ast::mutability) -> ast::mt {
    ast::mt {
        ty: ty,
        mutbl: mutbl
    }
}

pub fn mk_ty(cx: @ExtCtxt,
             span: span,
             ty: ast::ty_) -> @ast::Ty {
    @ast::Ty {
        id: cx.next_id(),
        span: span,
        node: ty
    }
}

pub fn mk_ty_path(cx: @ExtCtxt,
                  span: span,
                  idents: ~[ ast::ident ])
               -> @ast::Ty {
    let ty = build::mk_raw_path(span, idents);
    mk_ty_path_path(cx, span, ty)
}

pub fn mk_ty_path_global(cx: @ExtCtxt,
                         span: span,
                         idents: ~[ ast::ident ])
                      -> @ast::Ty {
    let ty = build::mk_raw_path_global(span, idents);
    mk_ty_path_path(cx, span, ty)
}

pub fn mk_ty_path_path(cx: @ExtCtxt,
                       span: span,
                       path: @ast::Path)
                      -> @ast::Ty {
    let ty = ast::ty_path(path, cx.next_id());
    mk_ty(cx, span, ty)
}

pub fn mk_ty_rptr(cx: @ExtCtxt,
                  span: span,
                  ty: @ast::Ty,
                  lifetime: Option<@ast::Lifetime>,
                  mutbl: ast::mutability)
               -> @ast::Ty {
    mk_ty(cx, span,
          ast::ty_rptr(lifetime, mk_ty_mt(ty, mutbl)))
}
pub fn mk_ty_uniq(cx: @ExtCtxt, span: span, ty: @ast::Ty) -> @ast::Ty {
    mk_ty(cx, span, ast::ty_uniq(mk_ty_mt(ty, ast::m_imm)))
}
pub fn mk_ty_box(cx: @ExtCtxt, span: span,
                 ty: @ast::Ty, mutbl: ast::mutability) -> @ast::Ty {
    mk_ty(cx, span, ast::ty_box(mk_ty_mt(ty, mutbl)))
}



pub fn mk_ty_infer(cx: @ExtCtxt, span: span) -> @ast::Ty {
    mk_ty(cx, span, ast::ty_infer)
}
pub fn mk_trait_ref_global(cx: @ExtCtxt,
                           span: span,
                           idents: ~[ ast::ident ])
    -> @ast::trait_ref
{
    mk_trait_ref_(cx, build::mk_raw_path_global(span, idents))
}
pub fn mk_trait_ref_(cx: @ExtCtxt, path: @ast::Path) -> @ast::trait_ref {
    @ast::trait_ref {
        path: path,
        ref_id: cx.next_id()
    }
}
pub fn mk_simple_ty_path(cx: @ExtCtxt,
                         span: span,
                         ident: ast::ident)
                      -> @ast::Ty {
    mk_ty_path(cx, span, ~[ ident ])
}
pub fn mk_arg(cx: @ExtCtxt,
              span: span,
              ident: ast::ident,
              ty: @ast::Ty)
           -> ast::arg {
    let arg_pat = mk_pat_ident(cx, span, ident);
    ast::arg {
        is_mutbl: false,
        ty: ty,
        pat: arg_pat,
        id: cx.next_id()
    }
}
pub fn mk_fn_decl(inputs: ~[ast::arg], output: @ast::Ty) -> ast::fn_decl {
    ast::fn_decl { inputs: inputs, output: output, cf: ast::return_val }
}
pub fn mk_trait_ty_param_bound_global(cx: @ExtCtxt,
                                      span: span,
                                      idents: ~[ast::ident])
                                   -> ast::TyParamBound {
    ast::TraitTyParamBound(mk_trait_ref_global(cx, span, idents))
}
pub fn mk_trait_ty_param_bound_(cx: @ExtCtxt,
                                path: @ast::Path) -> ast::TyParamBound {
    ast::TraitTyParamBound(mk_trait_ref_(cx, path))
}
pub fn mk_ty_param(cx: @ExtCtxt,
                   ident: ast::ident,
                   bounds: @OptVec<ast::TyParamBound>)
                -> ast::TyParam {
    ast::TyParam { ident: ident, id: cx.next_id(), bounds: bounds }
}
pub fn mk_lifetime(cx: @ExtCtxt,
                   span: span,
                   ident: ast::ident)
                -> ast::Lifetime {
    ast::Lifetime { id: cx.next_id(), span: span, ident: ident }
}
pub fn mk_arm(cx: @ExtCtxt,
              span: span,
              pats: ~[@ast::pat],
              expr: @ast::expr)
           -> ast::arm {
    ast::arm {
        pats: pats,
        guard: None,
        body: mk_simple_block(cx, span, expr)
    }
}
pub fn mk_unreachable(cx: @ExtCtxt, span: span) -> @ast::expr {
    let loc = cx.codemap().lookup_char_pos(span.lo);
    mk_call_global(
        cx,
        span,
        ~[
            cx.ident_of("core"),
            cx.ident_of("sys"),
            cx.ident_of("FailWithCause"),
            cx.ident_of("fail_with"),
        ],
        ~[
            mk_base_str(cx, span, ~"internal error: entered unreachable code"),
            mk_base_str(cx, span, copy loc.file.name),
            mk_uint(cx, span, loc.line),
        ]
    )
}
pub fn mk_unreachable_arm(cx: @ExtCtxt, span: span) -> ast::arm {
    mk_arm(cx, span, ~[mk_pat_wild(cx, span)], mk_unreachable(cx, span))
}

pub fn make_self(cx: @ExtCtxt, span: span) -> @ast::expr {
    build::mk_expr(cx, span, ast::expr_self)
}

//
// Duplication functions
//
// These functions just duplicate AST nodes.
//

pub fn duplicate_expr(cx: @ExtCtxt, expr: @ast::expr) -> @ast::expr {
    let folder = fold::default_ast_fold();
    let folder = @fold::AstFoldFns {
        new_id: |_| cx.next_id(),
        ..*folder
    };
    let folder = fold::make_fold(folder);
    folder.fold_expr(expr)
}



// Transitional reexports so qquote can find the paths it is looking for
mod syntax {
    pub use ext;
    pub use parse;
}

pub trait AstBuilder {
    // paths
    fn path(&self, span: span, strs: ~[ast::ident]) -> @ast::Path;
    fn path_global(&self, span: span, strs: ~[ast::ident]) -> @ast::Path;
    fn path_tps(&self, span: span, strs: ~[ast::ident], tps: ~[@ast::Ty])
                -> @ast::Path;
    fn path_tps_global(&self,
                       span: span,
                       strs: ~[ast::ident],
                       tps: ~[@ast::Ty])
                       -> @ast::Path;

    // types
    fn ty_path(&self, @ast::Path) -> @ast::Ty;

    fn ty_param(&self, id: ast::ident, bounds: @OptVec<ast::TyParamBound>)
        -> ast::TyParam;
    fn ty_vars(&self, ty_params: &OptVec<ast::TyParam>) -> ~[@ast::Ty];
    fn ty_vars_global(&self, ty_params: &OptVec<ast::TyParam>) -> ~[@ast::Ty];
    fn ty_field_imm(&self, name: ident, ty: @ast::Ty) -> ast::ty_field;
    fn ty_option(&self, ty: @ast::Ty) -> @ast::Ty;
    fn ty_infer(&self) -> @ast::Ty;
    fn ty_nil_ast_builder(&self) -> @ast::Ty;
    fn strip_bounds(&self, bounds: &Generics) -> Generics;


    // statements
    fn stmt_expr(&self, expr: @ast::expr) -> @ast::stmt;
    fn stmt_let(&self, ident: ident, e: @ast::expr) -> @ast::stmt;

    // literals
    fn lit_str(&self, span: span, s: @~str) -> @ast::expr;
    fn lit_uint(&self, span: span, i: uint) -> @ast::expr;

    // blocks
    fn blk(&self, span: span, stmts: ~[@ast::stmt], expr: Option<@ast::expr>) -> ast::blk;
    fn blk_expr(&self, expr: @ast::expr) -> ast::blk;

    // expressions
    fn expr(&self, span: span, node: ast::expr_) -> @ast::expr;
    fn expr_path(&self, span: span, strs: ~[ast::ident]) -> @ast::expr;
    fn expr_path_global(&self, span: span, strs: ~[ast::ident]) -> @ast::expr;
    fn expr_var(&self, span: span, var: &str) -> @ast::expr;
    fn expr_self(&self, span: span) -> @ast::expr;
    fn expr_field(&self, span: span, expr: @ast::expr, ident: ast::ident)
                  -> @ast::expr;
    fn expr_call(&self, span: span, expr: @ast::expr, args: ~[@ast::expr])
                 -> @ast::expr;
    fn expr_method_call(&self,
                        span: span,
                        expr: @ast::expr,
                        ident: ast::ident,
                        args: ~[@ast::expr])
                        -> @ast::expr;
    fn expr_blk(&self, b: ast::blk) -> @ast::expr;
    fn field_imm(&self, name: ident, e: @ast::expr) -> ast::field;
    fn expr_struct(&self,
                   path: @ast::Path,
                   fields: ~[ast::field]) -> @ast::expr;
    fn lambda0(&self, blk: ast::blk) -> @ast::expr;
    fn lambda1(&self, blk: ast::blk, ident: ast::ident) -> @ast::expr;
    fn lambda_expr_0(&self, expr: @ast::expr) -> @ast::expr;
    fn lambda_expr_1(&self, expr: @ast::expr, ident: ast::ident)
                    -> @ast::expr;
    fn lambda_stmts_0(&self, span: span, stmts: ~[@ast::stmt]) -> @ast::expr;
    fn lambda_stmts_1(&self,
                      span: span,
                      stmts: ~[@ast::stmt],
                      ident: ast::ident)
                      -> @ast::expr;

    // items
    fn item(&self, name: ident, span: span, node: ast::item_) -> @ast::item;

    fn arg(&self, name: ident, ty: @ast::Ty) -> ast::arg;
    fn fn_decl(&self, inputs: ~[ast::arg], output: @ast::Ty) -> ast::fn_decl;

    fn item_fn_poly(&self,
                    ame: ident,
                    inputs: ~[ast::arg],
                    output: @ast::Ty,
                    generics: Generics,
                    body: ast::blk) -> @ast::item;
    fn item_fn(&self,
               name: ident,
               inputs: ~[ast::arg],
               output: @ast::Ty,
               body: ast::blk) -> @ast::item;

    fn variant(&self,
               name: ident,
               span: span,
               tys: ~[@ast::Ty]) -> ast::variant;
    fn item_enum_poly(&self,
                      name: ident,
                      span: span,
                      enum_definition: ast::enum_def,
                      generics: Generics) -> @ast::item;
    fn item_enum(&self,
                 name: ident,
                 span: span,
                 enum_definition: ast::enum_def) -> @ast::item;

    fn item_struct_poly(&self,
                        name: ident,
                        span: span,
                        struct_def: ast::struct_def,
                        generics: Generics) -> @ast::item;
    fn item_struct(&self,
                   name: ident,
                   span: span,
                   struct_def: ast::struct_def) -> @ast::item;

    fn item_mod(&self,
                name: ident,
                span: span,
                items: ~[@ast::item]) -> @ast::item;

    fn item_ty_poly(&self,
                    name: ident,
                    span: span,
                    ty: @ast::Ty,
                    generics: Generics) -> @ast::item;
    fn item_ty(&self, name: ident, span: span, ty: @ast::Ty) -> @ast::item;

}

impl AstBuilder for @ExtCtxt {
    fn path(&self, span: span, strs: ~[ast::ident]) -> @ast::Path {
        @ast::Path {
            span: span,
            global: false,
            idents: strs,
            rp: None,
            types: ~[]
        }
    }

    fn path_global(&self, span: span, strs: ~[ast::ident]) -> @ast::Path {
        @ast::Path {
            span: span,
            global: true,
            idents: strs,
            rp: None,
            types: ~[]
        }
    }

    fn path_tps(
        &self,
        span: span,
        strs: ~[ast::ident],
        tps: ~[@ast::Ty]
    ) -> @ast::Path {
        @ast::Path {
            span: span,
            global: false,
            idents: strs,
            rp: None,
            types: tps
        }
    }

    fn path_tps_global(
        &self,
        span: span,
        strs: ~[ast::ident],
        tps: ~[@ast::Ty]
    ) -> @ast::Path {
        @ast::Path {
            span: span,
            global: true,
            idents: strs,
            rp: None,
            types: tps
        }
    }

    fn ty_path(&self, path: @ast::Path) -> @ast::Ty {
        build::mk_ty(*self, path.span,
                     ast::ty_path(path, self.next_id()))
    }

    fn ty_option(&self, ty: @ast::Ty) -> @ast::Ty {
        self.ty_path(
            self.path_tps_global(dummy_sp(),
                                 ~[
                                     self.ident_of("core"),
                                     self.ident_of("option"),
                                     self.ident_of("Option")
                                 ],
                                 ~[ ty ]))
    }

    fn ty_field_imm(&self, name: ident, ty: @ast::Ty) -> ast::ty_field {
        spanned {
            node: ast::ty_field_ {
                ident: name,
                mt: ast::mt { ty: ty, mutbl: ast::m_imm },
            },
            span: dummy_sp(),
        }
    }

    fn ty_infer(&self) -> @ast::Ty {
        @ast::Ty {
            id: self.next_id(),
            node: ast::ty_infer,
            span: dummy_sp(),
        }
    }

    fn ty_param(&self, id: ast::ident, bounds: @OptVec<ast::TyParamBound>)
        -> ast::TyParam
    {
        ast::TyParam { ident: id, id: self.next_id(), bounds: bounds }
    }

    fn ty_nil_ast_builder(&self) -> @ast::Ty {
        @ast::Ty {
            id: self.next_id(),
            node: ast::ty_nil,
            span: dummy_sp(),
        }
    }

    fn ty_vars(&self, ty_params: &OptVec<ast::TyParam>) -> ~[@ast::Ty] {
        opt_vec::take_vec(
            ty_params.map(|p| self.ty_path(
                mk_raw_path(dummy_sp(), ~[p.ident]))))
    }

    fn ty_vars_global(&self,
                      ty_params: &OptVec<ast::TyParam>) -> ~[@ast::Ty] {
        opt_vec::take_vec(
            ty_params.map(|p| self.ty_path(
                mk_raw_path(dummy_sp(), ~[p.ident]))))
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


    fn stmt_expr(&self, expr: @ast::expr) -> @ast::stmt {
        @codemap::spanned { node: ast::stmt_semi(expr, self.next_id()),
                       span: expr.span }
    }

    fn stmt_let(&self, ident: ident, e: @ast::expr) -> @ast::stmt {
        let ext_cx = *self;
        quote_stmt!( let $ident = $e; )
    }

    fn lit_str(&self, span: span, s: @~str) -> @ast::expr {
        self.expr(
            span,
            ast::expr_vstore(
                self.expr(
                    span,
                    ast::expr_lit(
                        @codemap::spanned { node: ast::lit_str(s),
                                        span: span})),
                ast::expr_vstore_uniq))
    }

    fn lit_uint(&self, span: span, i: uint) -> @ast::expr {
        self.expr(
            span,
            ast::expr_lit(
                @codemap::spanned { node: ast::lit_uint(i as u64, ast::ty_u),
                                span: span}))
    }

    fn blk(&self, span: span, stmts: ~[@ast::stmt], expr: Option<@expr>) -> ast::blk {
        codemap::spanned {
            node: ast::blk_ {
                view_items: ~[],
                stmts: stmts,
                expr: expr,
                id: self.next_id(),
                rules: ast::default_blk,
            },
            span: span,
        }
    }

    fn blk_expr(&self, expr: @ast::expr) -> ast::blk {
        self.blk(expr.span, ~[], Some(expr))
    }

    fn expr(&self, span: span, node: ast::expr_) -> @ast::expr {
        @ast::expr {
            id: self.next_id(),
            callee_id: self.next_id(),
            node: node,
            span: span,
        }
    }

    fn expr_path(&self, span: span, strs: ~[ast::ident]) -> @ast::expr {
        self.expr(span, ast::expr_path(self.path(span, strs)))
    }

    fn expr_path_global(
        &self,
        span: span,
        strs: ~[ast::ident]
    ) -> @ast::expr {
        self.expr(span, ast::expr_path(self.path_global(span, strs)))
    }

    fn expr_var(&self, span: span, var: &str) -> @ast::expr {
        self.expr_path(span, ~[self.ident_of(var)])
    }

    fn expr_self(&self, span: span) -> @ast::expr {
        self.expr(span, ast::expr_self)
    }

    fn expr_field(
        &self,
        span: span,
        expr: @ast::expr,
        ident: ast::ident
    ) -> @ast::expr {
        self.expr(span, ast::expr_field(expr, ident, ~[]))
    }

    fn expr_call(
        &self,
        span: span,
        expr: @ast::expr,
        args: ~[@ast::expr]
    ) -> @ast::expr {
        self.expr(span, ast::expr_call(expr, args, ast::NoSugar))
    }

    fn expr_method_call(
        &self,
        span: span,
        expr: @ast::expr,
        ident: ast::ident,
        args: ~[@ast::expr]
    ) -> @ast::expr {
        self.expr(span,
                  ast::expr_method_call(expr, ident, ~[], args, ast::NoSugar))
    }
    fn expr_blk(&self, b: ast::blk) -> @ast::expr {
        self.expr(dummy_sp(), ast::expr_block(b))
    }
    fn field_imm(&self, name: ident, e: @ast::expr) -> ast::field {
        spanned {
            node: ast::field_ { mutbl: ast::m_imm, ident: name, expr: e },
            span: dummy_sp(),
        }
    }
    fn expr_struct(&self, path: @ast::Path,
                   fields: ~[ast::field]) -> @ast::expr {
        @ast::expr {
            id: self.next_id(),
            callee_id: self.next_id(),
            node: ast::expr_struct(path, fields, None),
            span: dummy_sp()
        }
    }


    fn lambda0(&self, blk: ast::blk) -> @ast::expr {
        let ext_cx = *self;
        let blk_e = self.expr(copy blk.span, ast::expr_block(copy blk));
        quote_expr!( || $blk_e )
    }

    fn lambda1(&self, blk: ast::blk, ident: ast::ident) -> @ast::expr {
        let ext_cx = *self;
        let blk_e = self.expr(copy blk.span, ast::expr_block(copy blk));
        quote_expr!( |$ident| $blk_e )
    }

    fn lambda_expr_0(&self, expr: @ast::expr) -> @ast::expr {
        self.lambda0(self.blk_expr(expr))
    }

    fn lambda_expr_1(&self, expr: @ast::expr, ident: ast::ident)
                    -> @ast::expr {
        self.lambda1(self.blk_expr(expr), ident)
    }

    fn lambda_stmts_0(&self, span: span, stmts: ~[@ast::stmt]) -> @ast::expr {
        self.lambda0(self.blk(span, stmts, None))
    }

    fn lambda_stmts_1(&self,
                      span: span,
                      stmts: ~[@ast::stmt],
                      ident: ast::ident)
                      -> @ast::expr {
        self.lambda1(self.blk(span, stmts, None), ident)
    }


    fn arg(&self, name: ident, ty: @ast::Ty) -> ast::arg {
        ast::arg {
            is_mutbl: false,
            ty: ty,
            pat: @ast::pat {
                id: self.next_id(),
                node: ast::pat_ident(
                    ast::bind_by_copy,
                    ast_util::ident_to_path(dummy_sp(), name),
                    None),
                span: dummy_sp(),
            },
            id: self.next_id(),
        }
    }

    fn fn_decl(&self, inputs: ~[ast::arg],
               output: @ast::Ty) -> ast::fn_decl {
        ast::fn_decl {
            inputs: inputs,
            output: output,
            cf: ast::return_val,
        }
    }

    fn item(&self, name: ident, span: span,
            node: ast::item_) -> @ast::item {

        // XXX: Would be nice if our generated code didn't violate
        // Rust coding conventions
        let non_camel_case_attribute = respan(dummy_sp(), ast::attribute_ {
            style: ast::attr_outer,
            value: @respan(dummy_sp(),
                          ast::meta_list(@~"allow", ~[
                              @respan(dummy_sp(),
                                      ast::meta_word(
                                         @~"non_camel_case_types"))
                          ])),
            is_sugared_doc: false
        });

        @ast::item { ident: name,
                     attrs: ~[non_camel_case_attribute],
                     id: self.next_id(),
                     node: node,
                     vis: ast::public,
                     span: span }
    }

    fn item_fn_poly(&self, name: ident,
                    inputs: ~[ast::arg],
                    output: @ast::Ty,
                    generics: Generics,
                    body: ast::blk) -> @ast::item {
        self.item(name,
                  dummy_sp(),
                  ast::item_fn(self.fn_decl(inputs, output),
                               ast::impure_fn,
                               AbiSet::Rust(),
                               generics,
                               body))
    }

    fn item_fn(&self,
               name: ident,
               inputs: ~[ast::arg],
               output: @ast::Ty,
               body: ast::blk
    ) -> @ast::item {
        self.item_fn_poly(
            name,
            inputs,
            output,
            ast_util::empty_generics(),
            body
        )
    }

    fn variant(&self, name: ident, span: span,
               tys: ~[@ast::Ty]) -> ast::variant {
        let args = do tys.map |ty| {
            ast::variant_arg { ty: *ty, id: self.next_id() }
        };

        spanned {
            node: ast::variant_ {
                name: name,
                attrs: ~[],
                kind: ast::tuple_variant_kind(args),
                id: self.next_id(),
                disr_expr: None,
                vis: ast::public
            },
            span: span,
        }
    }

    fn item_enum_poly(&self, name: ident, span: span,
                      enum_definition: ast::enum_def,
                      generics: Generics) -> @ast::item {
        self.item(name, span, ast::item_enum(enum_definition, generics))
    }

    fn item_enum(&self, name: ident, span: span,
                 enum_definition: ast::enum_def) -> @ast::item {
        self.item_enum_poly(name, span, enum_definition,
                            ast_util::empty_generics())
    }

    fn item_struct(
        &self, name: ident,
        span: span,
        struct_def: ast::struct_def
    ) -> @ast::item {
        self.item_struct_poly(
            name,
            span,
            struct_def,
            ast_util::empty_generics()
        )
    }

    fn item_struct_poly(
        &self,
        name: ident,
        span: span,
        struct_def: ast::struct_def,
        generics: Generics
    ) -> @ast::item {
        self.item(name, span, ast::item_struct(@struct_def, generics))
    }

    fn item_mod(&self, name: ident, span: span,
                items: ~[@ast::item]) -> @ast::item {

        // XXX: Total hack: import `core::kinds::Owned` to work around a
        // parser bug whereby `fn f<T:::kinds::Owned>` doesn't parse.
        let vi = ast::view_item_use(~[
            @codemap::spanned {
                node: ast::view_path_simple(
                    self.ident_of("Owned"),
                    mk_raw_path(
                        codemap::dummy_sp(),
                        ~[
                            self.ident_of("core"),
                            self.ident_of("kinds"),
                            self.ident_of("Owned")
                        ]
                    ),
                    self.next_id()
                ),
                span: codemap::dummy_sp()
            }
        ]);
        let vi = @ast::view_item {
            node: vi,
            attrs: ~[],
            vis: ast::private,
            span: codemap::dummy_sp()
        };

        self.item(
            name,
            span,
            ast::item_mod(ast::_mod {
                view_items: ~[vi],
                items: items,
            })
        )
    }

    fn item_ty_poly(&self, name: ident, span: span, ty: @ast::Ty,
                    generics: Generics) -> @ast::item {
        self.item(name, span, ast::item_ty(ty, generics))
    }

    fn item_ty(&self, name: ident, span: span, ty: @ast::Ty) -> @ast::item {
        self.item_ty_poly(name, span, ty, ast_util::empty_generics())
    }


}
