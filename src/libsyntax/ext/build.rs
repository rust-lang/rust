// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use codemap;
use codemap::span;
use fold;
use ext::base::ext_ctxt;
use ext::build;

use opt_vec::OptVec;

pub struct Field {
    ident: ast::ident,
    ex: @ast::expr
}

pub fn mk_expr(cx: @ext_ctxt,
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

pub fn mk_lit(cx: @ext_ctxt, sp: span, lit: ast::lit_) -> @ast::expr {
    let sp_lit = @codemap::spanned { node: lit, span: sp };
    mk_expr(cx, sp, ast::expr_lit(sp_lit))
}
pub fn mk_int(cx: @ext_ctxt, sp: span, i: int) -> @ast::expr {
    let lit = ast::lit_int(i as i64, ast::ty_i);
    return mk_lit(cx, sp, lit);
}
pub fn mk_uint(cx: @ext_ctxt, sp: span, u: uint) -> @ast::expr {
    let lit = ast::lit_uint(u as u64, ast::ty_u);
    return mk_lit(cx, sp, lit);
}
pub fn mk_u8(cx: @ext_ctxt, sp: span, u: u8) -> @ast::expr {
    let lit = ast::lit_uint(u as u64, ast::ty_u8);
    return mk_lit(cx, sp, lit);
}
pub fn mk_binary(cx: @ext_ctxt, sp: span, op: ast::binop,
                 lhs: @ast::expr, rhs: @ast::expr) -> @ast::expr {
    cx.next_id(); // see ast_util::op_expr_callee_id
    mk_expr(cx, sp, ast::expr_binary(op, lhs, rhs))
}

pub fn mk_deref(cx: @ext_ctxt, sp: span, e: @ast::expr) -> @ast::expr {
    mk_unary(cx, sp, ast::deref, e)
}
pub fn mk_unary(cx: @ext_ctxt, sp: span, op: ast::unop, e: @ast::expr)
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
pub fn mk_path_raw(cx: @ext_ctxt, sp: span, path: @ast::Path)-> @ast::expr {
    mk_expr(cx, sp, ast::expr_path(path))
}
pub fn mk_path(cx: @ext_ctxt, sp: span, idents: ~[ast::ident])
            -> @ast::expr {
    mk_path_raw(cx, sp, mk_raw_path(sp, idents))
}
pub fn mk_path_global(cx: @ext_ctxt, sp: span, idents: ~[ast::ident])
                   -> @ast::expr {
    mk_path_raw(cx, sp, mk_raw_path_global(sp, idents))
}
pub fn mk_access_(cx: @ext_ctxt, sp: span, p: @ast::expr, m: ast::ident)
               -> @ast::expr {
    mk_expr(cx, sp, ast::expr_field(p, m, ~[]))
}
pub fn mk_access(cx: @ext_ctxt, sp: span, p: ~[ast::ident], m: ast::ident)
              -> @ast::expr {
    let pathexpr = mk_path(cx, sp, p);
    return mk_access_(cx, sp, pathexpr, m);
}
pub fn mk_addr_of(cx: @ext_ctxt, sp: span, e: @ast::expr) -> @ast::expr {
    return mk_expr(cx, sp, ast::expr_addr_of(ast::m_imm, e));
}
pub fn mk_mut_addr_of(cx: @ext_ctxt, sp: span, e: @ast::expr) -> @ast::expr {
    return mk_expr(cx, sp, ast::expr_addr_of(ast::m_mutbl, e));
}
pub fn mk_method_call(cx: @ext_ctxt,
                      sp: span,
                      rcvr_expr: @ast::expr,
                      method_ident: ast::ident,
                      args: ~[@ast::expr]) -> @ast::expr {
    mk_expr(cx, sp, ast::expr_method_call(rcvr_expr, method_ident, ~[], args, ast::NoSugar))
}
pub fn mk_call_(cx: @ext_ctxt, sp: span, fn_expr: @ast::expr,
                args: ~[@ast::expr]) -> @ast::expr {
    mk_expr(cx, sp, ast::expr_call(fn_expr, args, ast::NoSugar))
}
pub fn mk_call(cx: @ext_ctxt, sp: span, fn_path: ~[ast::ident],
               args: ~[@ast::expr]) -> @ast::expr {
    let pathexpr = mk_path(cx, sp, fn_path);
    return mk_call_(cx, sp, pathexpr, args);
}
pub fn mk_call_global(cx: @ext_ctxt, sp: span, fn_path: ~[ast::ident],
                      args: ~[@ast::expr]) -> @ast::expr {
    let pathexpr = mk_path_global(cx, sp, fn_path);
    return mk_call_(cx, sp, pathexpr, args);
}
// e = expr, t = type
pub fn mk_base_vec_e(cx: @ext_ctxt, sp: span, exprs: ~[@ast::expr])
                  -> @ast::expr {
    let vecexpr = ast::expr_vec(exprs, ast::m_imm);
    mk_expr(cx, sp, vecexpr)
}
pub fn mk_vstore_e(cx: @ext_ctxt, sp: span, expr: @ast::expr,
                   vst: ast::expr_vstore) ->
   @ast::expr {
    mk_expr(cx, sp, ast::expr_vstore(expr, vst))
}
pub fn mk_uniq_vec_e(cx: @ext_ctxt, sp: span, exprs: ~[@ast::expr])
                  -> @ast::expr {
    mk_vstore_e(cx, sp, mk_base_vec_e(cx, sp, exprs), ast::expr_vstore_uniq)
}
pub fn mk_slice_vec_e(cx: @ext_ctxt, sp: span, exprs: ~[@ast::expr])
                   -> @ast::expr {
    mk_vstore_e(cx, sp, mk_base_vec_e(cx, sp, exprs),
                ast::expr_vstore_slice)
}
pub fn mk_base_str(cx: @ext_ctxt, sp: span, s: ~str) -> @ast::expr {
    let lit = ast::lit_str(@s);
    return mk_lit(cx, sp, lit);
}
pub fn mk_uniq_str(cx: @ext_ctxt, sp: span, s: ~str) -> @ast::expr {
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
pub fn mk_struct_e(cx: @ext_ctxt,
                   sp: span,
                   ctor_path: ~[ast::ident],
                   fields: ~[Field])
                -> @ast::expr {
    mk_expr(cx, sp,
            ast::expr_struct(mk_raw_path(sp, ctor_path),
                             mk_fields(sp, fields),
                                    option::None::<@ast::expr>))
}
pub fn mk_global_struct_e(cx: @ext_ctxt,
                          sp: span,
                          ctor_path: ~[ast::ident],
                          fields: ~[Field])
                       -> @ast::expr {
    mk_expr(cx, sp,
            ast::expr_struct(mk_raw_path_global(sp, ctor_path),
                             mk_fields(sp, fields),
                                    option::None::<@ast::expr>))
}
pub fn mk_glob_use(cx: @ext_ctxt,
                   sp: span,
                   path: ~[ast::ident]) -> @ast::view_item {
    let glob = @codemap::spanned {
        node: ast::view_path_glob(mk_raw_path(sp, path), cx.next_id()),
        span: sp,
    };
    @ast::view_item { node: ast::view_item_use(~[glob]),
                      attrs: ~[],
                      vis: ast::private,
                      span: sp }
}
pub fn mk_local(cx: @ext_ctxt, sp: span, mutbl: bool,
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
pub fn mk_block(cx: @ext_ctxt, span: span,
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
pub fn mk_block_(cx: @ext_ctxt,
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
pub fn mk_simple_block(cx: @ext_ctxt,
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
pub fn mk_lambda_(cx: @ext_ctxt,
                 span: span,
                 fn_decl: ast::fn_decl,
                 blk: ast::blk)
              -> @ast::expr {
    mk_expr(cx, span, ast::expr_fn_block(fn_decl, blk))
}
pub fn mk_lambda(cx: @ext_ctxt,
                 span: span,
                 fn_decl: ast::fn_decl,
                 expr: @ast::expr)
              -> @ast::expr {
    let blk = mk_simple_block(cx, span, expr);
    mk_lambda_(cx, span, fn_decl, blk)
}
pub fn mk_lambda_stmts(cx: @ext_ctxt,
                       span: span,
                       fn_decl: ast::fn_decl,
                       stmts: ~[@ast::stmt])
                    -> @ast::expr {
    let blk = mk_block(cx, span, ~[], stmts, None);
    mk_lambda(cx, span, fn_decl, blk)
}
pub fn mk_lambda_no_args(cx: @ext_ctxt,
                         span: span,
                         expr: @ast::expr)
                      -> @ast::expr {
    let fn_decl = mk_fn_decl(~[], mk_ty_infer(cx, span));
    mk_lambda(cx, span, fn_decl, expr)
}
pub fn mk_copy(cx: @ext_ctxt, sp: span, e: @ast::expr) -> @ast::expr {
    mk_expr(cx, sp, ast::expr_copy(e))
}
pub fn mk_managed(cx: @ext_ctxt, sp: span, e: @ast::expr) -> @ast::expr {
    mk_expr(cx, sp, ast::expr_unary(ast::box(ast::m_imm), e))
}
pub fn mk_pat(cx: @ext_ctxt, span: span, pat: ast::pat_) -> @ast::pat {
    @ast::pat { id: cx.next_id(), node: pat, span: span }
}
pub fn mk_pat_wild(cx: @ext_ctxt, span: span) -> @ast::pat {
    mk_pat(cx, span, ast::pat_wild)
}
pub fn mk_pat_lit(cx: @ext_ctxt,
                  span: span,
                  expr: @ast::expr) -> @ast::pat {
    mk_pat(cx, span, ast::pat_lit(expr))
}
pub fn mk_pat_ident(cx: @ext_ctxt,
                    span: span,
                    ident: ast::ident) -> @ast::pat {
    mk_pat_ident_with_binding_mode(cx, span, ident, ast::bind_by_copy)
}

pub fn mk_pat_ident_with_binding_mode(cx: @ext_ctxt,
                                      span: span,
                                      ident: ast::ident,
                                      bm: ast::binding_mode) -> @ast::pat {
    let path = mk_raw_path(span, ~[ ident ]);
    let pat = ast::pat_ident(bm, path, None);
    mk_pat(cx, span, pat)
}
pub fn mk_pat_enum(cx: @ext_ctxt,
                   span: span,
                   path: @ast::Path,
                   subpats: ~[@ast::pat])
                -> @ast::pat {
    let pat = ast::pat_enum(path, Some(subpats));
    mk_pat(cx, span, pat)
}
pub fn mk_pat_struct(cx: @ext_ctxt,
                     span: span,
                     path: @ast::Path,
                     field_pats: ~[ast::field_pat])
                  -> @ast::pat {
    let pat = ast::pat_struct(path, field_pats, false);
    mk_pat(cx, span, pat)
}
pub fn mk_bool(cx: @ext_ctxt, span: span, value: bool) -> @ast::expr {
    let lit_expr = ast::expr_lit(@codemap::spanned {
        node: ast::lit_bool(value),
        span: span });
    build::mk_expr(cx, span, lit_expr)
}
pub fn mk_stmt(cx: @ext_ctxt, span: span, expr: @ast::expr) -> @ast::stmt {
    let stmt_ = ast::stmt_semi(expr, cx.next_id());
    @codemap::spanned { node: stmt_, span: span }
}

pub fn mk_ty_mt(ty: @ast::Ty, mutbl: ast::mutability) -> ast::mt {
    ast::mt {
        ty: ty,
        mutbl: mutbl
    }
}

pub fn mk_ty(cx: @ext_ctxt,
             span: span,
             ty: ast::ty_) -> @ast::Ty {
    @ast::Ty {
        id: cx.next_id(),
        span: span,
        node: ty
    }
}

pub fn mk_ty_path(cx: @ext_ctxt,
                  span: span,
                  idents: ~[ ast::ident ])
               -> @ast::Ty {
    let ty = build::mk_raw_path(span, idents);
    mk_ty_path_path(cx, span, ty)
}

pub fn mk_ty_path_global(cx: @ext_ctxt,
                         span: span,
                         idents: ~[ ast::ident ])
                      -> @ast::Ty {
    let ty = build::mk_raw_path_global(span, idents);
    mk_ty_path_path(cx, span, ty)
}

pub fn mk_ty_path_path(cx: @ext_ctxt,
                       span: span,
                       path: @ast::Path)
                      -> @ast::Ty {
    let ty = ast::ty_path(path, cx.next_id());
    mk_ty(cx, span, ty)
}

pub fn mk_ty_rptr(cx: @ext_ctxt,
                  span: span,
                  ty: @ast::Ty,
                  lifetime: Option<@ast::Lifetime>,
                  mutbl: ast::mutability)
               -> @ast::Ty {
    mk_ty(cx, span,
          ast::ty_rptr(lifetime, mk_ty_mt(ty, mutbl)))
}
pub fn mk_ty_uniq(cx: @ext_ctxt, span: span, ty: @ast::Ty) -> @ast::Ty {
    mk_ty(cx, span, ast::ty_uniq(mk_ty_mt(ty, ast::m_imm)))
}
pub fn mk_ty_box(cx: @ext_ctxt, span: span,
                 ty: @ast::Ty, mutbl: ast::mutability) -> @ast::Ty {
    mk_ty(cx, span, ast::ty_box(mk_ty_mt(ty, mutbl)))
}



pub fn mk_ty_infer(cx: @ext_ctxt, span: span) -> @ast::Ty {
    mk_ty(cx, span, ast::ty_infer)
}
pub fn mk_trait_ref_global(cx: @ext_ctxt,
                           span: span,
                           idents: ~[ ast::ident ])
    -> @ast::trait_ref
{
    mk_trait_ref_(cx, build::mk_raw_path_global(span, idents))
}
pub fn mk_trait_ref_(cx: @ext_ctxt, path: @ast::Path) -> @ast::trait_ref {
    @ast::trait_ref {
        path: path,
        ref_id: cx.next_id()
    }
}
pub fn mk_simple_ty_path(cx: @ext_ctxt,
                         span: span,
                         ident: ast::ident)
                      -> @ast::Ty {
    mk_ty_path(cx, span, ~[ ident ])
}
pub fn mk_arg(cx: @ext_ctxt,
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
pub fn mk_trait_ty_param_bound_global(cx: @ext_ctxt,
                                      span: span,
                                      idents: ~[ast::ident])
                                   -> ast::TyParamBound {
    ast::TraitTyParamBound(mk_trait_ref_global(cx, span, idents))
}
pub fn mk_trait_ty_param_bound_(cx: @ext_ctxt,
                                path: @ast::Path) -> ast::TyParamBound {
    ast::TraitTyParamBound(mk_trait_ref_(cx, path))
}
pub fn mk_ty_param(cx: @ext_ctxt,
                   ident: ast::ident,
                   bounds: @OptVec<ast::TyParamBound>)
                -> ast::TyParam {
    ast::TyParam { ident: ident, id: cx.next_id(), bounds: bounds }
}
pub fn mk_lifetime(cx: @ext_ctxt,
                   span: span,
                   ident: ast::ident)
                -> ast::Lifetime {
    ast::Lifetime { id: cx.next_id(), span: span, ident: ident }
}
pub fn mk_arm(cx: @ext_ctxt,
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
pub fn mk_unreachable(cx: @ext_ctxt, span: span) -> @ast::expr {
    let loc = cx.codemap().lookup_char_pos(span.lo);
    mk_call_global(
        cx,
        span,
        ~[
            cx.ident_of(~"core"),
            cx.ident_of(~"sys"),
            cx.ident_of(~"FailWithCause"),
            cx.ident_of(~"fail_with"),
        ],
        ~[
            mk_base_str(cx, span, ~"internal error: entered unreachable code"),
            mk_base_str(cx, span, loc.file.name),
            mk_uint(cx, span, loc.line),
        ]
    )
}
pub fn mk_unreachable_arm(cx: @ext_ctxt, span: span) -> ast::arm {
    mk_arm(cx, span, ~[mk_pat_wild(cx, span)], mk_unreachable(cx, span))
}

//
// Duplication functions
//
// These functions just duplicate AST nodes.
//

pub fn duplicate_expr(cx: @ext_ctxt, expr: @ast::expr) -> @ast::expr {
    let folder = fold::default_ast_fold();
    let folder = @fold::AstFoldFns {
        new_id: |_| cx.next_id(),
        ..*folder
    };
    let folder = fold::make_fold(folder);
    folder.fold_expr(expr)
}

