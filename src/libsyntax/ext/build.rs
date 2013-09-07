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
use ast::Ident;
use ast;
use ast_util;
use codemap::{Span, respan, dummy_sp};
use ext::base::ExtCtxt;
use ext::quote::rt::*;
use opt_vec;
use opt_vec::OptVec;

pub struct Field {
    ident: ast::Ident,
    ex: @ast::Expr
}

// Transitional reexports so qquote can find the paths it is looking for
mod syntax {
    pub use ext;
    pub use parse;
}

pub trait AstBuilder {
    // paths
    fn path(&self, span: Span, strs: ~[ast::Ident]) -> ast::Path;
    fn path_ident(&self, span: Span, id: ast::Ident) -> ast::Path;
    fn path_global(&self, span: Span, strs: ~[ast::Ident]) -> ast::Path;
    fn path_all(&self, sp: Span,
                global: bool,
                idents: ~[ast::Ident],
                rp: Option<ast::Lifetime>,
                types: ~[ast::Ty])
        -> ast::Path;

    // types
    fn ty_mt(&self, ty: ast::Ty, mutbl: ast::Mutability) -> ast::mt;

    fn ty(&self, span: Span, ty: ast::ty_) -> ast::Ty;
    fn ty_path(&self, ast::Path, Option<OptVec<ast::TyParamBound>>) -> ast::Ty;
    fn ty_ident(&self, span: Span, idents: ast::Ident) -> ast::Ty;

    fn ty_rptr(&self, span: Span,
               ty: ast::Ty,
               lifetime: Option<ast::Lifetime>,
               mutbl: ast::Mutability) -> ast::Ty;
    fn ty_uniq(&self, span: Span, ty: ast::Ty) -> ast::Ty;
    fn ty_box(&self, span: Span, ty: ast::Ty, mutbl: ast::Mutability) -> ast::Ty;

    fn ty_option(&self, ty: ast::Ty) -> ast::Ty;
    fn ty_infer(&self, sp: Span) -> ast::Ty;
    fn ty_nil(&self) -> ast::Ty;

    fn ty_vars(&self, ty_params: &OptVec<ast::TyParam>) -> ~[ast::Ty];
    fn ty_vars_global(&self, ty_params: &OptVec<ast::TyParam>) -> ~[ast::Ty];
    fn ty_field_imm(&self, span: Span, name: Ident, ty: ast::Ty) -> ast::TypeField;
    fn strip_bounds(&self, bounds: &Generics) -> Generics;

    fn typaram(&self, id: ast::Ident, bounds: OptVec<ast::TyParamBound>) -> ast::TyParam;

    fn trait_ref(&self, path: ast::Path) -> ast::trait_ref;
    fn typarambound(&self, path: ast::Path) -> ast::TyParamBound;
    fn lifetime(&self, span: Span, ident: ast::Ident) -> ast::Lifetime;

    // statements
    fn stmt_expr(&self, expr: @ast::Expr) -> @ast::Stmt;
    fn stmt_let(&self, sp: Span, mutbl: bool, ident: ast::Ident, ex: @ast::Expr) -> @ast::Stmt;
    fn stmt_let_typed(&self,
                      sp: Span,
                      mutbl: bool,
                      ident: ast::Ident,
                      typ: ast::Ty,
                      ex: @ast::Expr)
                      -> @ast::Stmt;

    // blocks
    fn block(&self, span: Span, stmts: ~[@ast::Stmt], expr: Option<@ast::Expr>) -> ast::Block;
    fn block_expr(&self, expr: @ast::Expr) -> ast::Block;
    fn block_all(&self, span: Span,
                 view_items: ~[ast::view_item],
                 stmts: ~[@ast::Stmt],
                 expr: Option<@ast::Expr>) -> ast::Block;

    // expressions
    fn expr(&self, span: Span, node: ast::Expr_) -> @ast::Expr;
    fn expr_path(&self, path: ast::Path) -> @ast::Expr;
    fn expr_ident(&self, span: Span, id: ast::Ident) -> @ast::Expr;

    fn expr_self(&self, span: Span) -> @ast::Expr;
    fn expr_binary(&self, sp: Span, op: ast::BinOp,
                   lhs: @ast::Expr, rhs: @ast::Expr) -> @ast::Expr;
    fn expr_deref(&self, sp: Span, e: @ast::Expr) -> @ast::Expr;
    fn expr_unary(&self, sp: Span, op: ast::UnOp, e: @ast::Expr) -> @ast::Expr;

    fn expr_managed(&self, sp: Span, e: @ast::Expr) -> @ast::Expr;
    fn expr_addr_of(&self, sp: Span, e: @ast::Expr) -> @ast::Expr;
    fn expr_mut_addr_of(&self, sp: Span, e: @ast::Expr) -> @ast::Expr;
    fn expr_field_access(&self, span: Span, expr: @ast::Expr, ident: ast::Ident) -> @ast::Expr;
    fn expr_call(&self, span: Span, expr: @ast::Expr, args: ~[@ast::Expr]) -> @ast::Expr;
    fn expr_call_ident(&self, span: Span, id: ast::Ident, args: ~[@ast::Expr]) -> @ast::Expr;
    fn expr_call_global(&self, sp: Span, fn_path: ~[ast::Ident],
                        args: ~[@ast::Expr]) -> @ast::Expr;
    fn expr_method_call(&self, span: Span,
                        expr: @ast::Expr, ident: ast::Ident,
                        args: ~[@ast::Expr]) -> @ast::Expr;
    fn expr_block(&self, b: ast::Block) -> @ast::Expr;

    fn field_imm(&self, span: Span, name: Ident, e: @ast::Expr) -> ast::Field;
    fn expr_struct(&self, span: Span, path: ast::Path, fields: ~[ast::Field]) -> @ast::Expr;
    fn expr_struct_ident(&self, span: Span, id: ast::Ident, fields: ~[ast::Field]) -> @ast::Expr;

    fn expr_lit(&self, sp: Span, lit: ast::lit_) -> @ast::Expr;

    fn expr_uint(&self, span: Span, i: uint) -> @ast::Expr;
    fn expr_int(&self, sp: Span, i: int) -> @ast::Expr;
    fn expr_u8(&self, sp: Span, u: u8) -> @ast::Expr;
    fn expr_bool(&self, sp: Span, value: bool) -> @ast::Expr;

    fn expr_vstore(&self, sp: Span, expr: @ast::Expr, vst: ast::ExprVstore) -> @ast::Expr;
    fn expr_vec(&self, sp: Span, exprs: ~[@ast::Expr]) -> @ast::Expr;
    fn expr_vec_uniq(&self, sp: Span, exprs: ~[@ast::Expr]) -> @ast::Expr;
    fn expr_vec_slice(&self, sp: Span, exprs: ~[@ast::Expr]) -> @ast::Expr;
    fn expr_str(&self, sp: Span, s: @str) -> @ast::Expr;
    fn expr_str_uniq(&self, sp: Span, s: @str) -> @ast::Expr;

    fn expr_unreachable(&self, span: Span) -> @ast::Expr;

    fn pat(&self, span: Span, pat: ast::Pat_) -> @ast::Pat;
    fn pat_wild(&self, span: Span) -> @ast::Pat;
    fn pat_lit(&self, span: Span, expr: @ast::Expr) -> @ast::Pat;
    fn pat_ident(&self, span: Span, ident: ast::Ident) -> @ast::Pat;

    fn pat_ident_binding_mode(&self,
                              span: Span,
                              ident: ast::Ident,
                              bm: ast::BindingMode) -> @ast::Pat;
    fn pat_enum(&self, span: Span, path: ast::Path, subpats: ~[@ast::Pat]) -> @ast::Pat;
    fn pat_struct(&self, span: Span,
                  path: ast::Path, field_pats: ~[ast::FieldPat]) -> @ast::Pat;

    fn arm(&self, span: Span, pats: ~[@ast::Pat], expr: @ast::Expr) -> ast::Arm;
    fn arm_unreachable(&self, span: Span) -> ast::Arm;

    fn expr_match(&self, span: Span, arg: @ast::Expr, arms: ~[ast::Arm]) -> @ast::Expr;
    fn expr_if(&self, span: Span,
               cond: @ast::Expr, then: @ast::Expr, els: Option<@ast::Expr>) -> @ast::Expr;

    fn lambda_fn_decl(&self, span: Span, fn_decl: ast::fn_decl, blk: ast::Block) -> @ast::Expr;

    fn lambda(&self, span: Span, ids: ~[ast::Ident], blk: ast::Block) -> @ast::Expr;
    fn lambda0(&self, span: Span, blk: ast::Block) -> @ast::Expr;
    fn lambda1(&self, span: Span, blk: ast::Block, ident: ast::Ident) -> @ast::Expr;

    fn lambda_expr(&self, span: Span, ids: ~[ast::Ident], blk: @ast::Expr) -> @ast::Expr;
    fn lambda_expr_0(&self, span: Span, expr: @ast::Expr) -> @ast::Expr;
    fn lambda_expr_1(&self, span: Span, expr: @ast::Expr, ident: ast::Ident) -> @ast::Expr;

    fn lambda_stmts(&self, span: Span, ids: ~[ast::Ident], blk: ~[@ast::Stmt]) -> @ast::Expr;
    fn lambda_stmts_0(&self, span: Span, stmts: ~[@ast::Stmt]) -> @ast::Expr;
    fn lambda_stmts_1(&self, span: Span, stmts: ~[@ast::Stmt], ident: ast::Ident) -> @ast::Expr;

    // items
    fn item(&self, span: Span,
            name: Ident, attrs: ~[ast::Attribute], node: ast::item_) -> @ast::item;

    fn arg(&self, span: Span, name: Ident, ty: ast::Ty) -> ast::arg;
    // XXX unused self
    fn fn_decl(&self, inputs: ~[ast::arg], output: ast::Ty) -> ast::fn_decl;

    fn item_fn_poly(&self,
                    span: Span,
                    name: Ident,
                    inputs: ~[ast::arg],
                    output: ast::Ty,
                    generics: Generics,
                    body: ast::Block) -> @ast::item;
    fn item_fn(&self,
               span: Span,
               name: Ident,
               inputs: ~[ast::arg],
               output: ast::Ty,
               body: ast::Block) -> @ast::item;

    fn variant(&self, span: Span, name: Ident, tys: ~[ast::Ty]) -> ast::variant;
    fn item_enum_poly(&self,
                      span: Span,
                      name: Ident,
                      enum_definition: ast::enum_def,
                      generics: Generics) -> @ast::item;
    fn item_enum(&self, span: Span, name: Ident, enum_def: ast::enum_def) -> @ast::item;

    fn item_struct_poly(&self,
                        span: Span,
                        name: Ident,
                        struct_def: ast::struct_def,
                        generics: Generics) -> @ast::item;
    fn item_struct(&self, span: Span, name: Ident, struct_def: ast::struct_def) -> @ast::item;

    fn item_mod(&self, span: Span,
                name: Ident, attrs: ~[ast::Attribute],
                vi: ~[ast::view_item], items: ~[@ast::item]) -> @ast::item;

    fn item_ty_poly(&self,
                    span: Span,
                    name: Ident,
                    ty: ast::Ty,
                    generics: Generics) -> @ast::item;
    fn item_ty(&self, span: Span, name: Ident, ty: ast::Ty) -> @ast::item;

    fn attribute(&self, sp: Span, mi: @ast::MetaItem) -> ast::Attribute;

    fn meta_word(&self, sp: Span, w: @str) -> @ast::MetaItem;
    fn meta_list(&self, sp: Span, name: @str, mis: ~[@ast::MetaItem]) -> @ast::MetaItem;
    fn meta_name_value(&self, sp: Span, name: @str, value: ast::lit_) -> @ast::MetaItem;

    fn view_use(&self, sp: Span,
                vis: ast::visibility, vp: ~[@ast::view_path]) -> ast::view_item;
    fn view_use_list(&self, sp: Span, vis: ast::visibility,
                     path: ~[ast::Ident], imports: &[ast::Ident]) -> ast::view_item;
    fn view_use_glob(&self, sp: Span,
                     vis: ast::visibility, path: ~[ast::Ident]) -> ast::view_item;
}

impl AstBuilder for @ExtCtxt {
    fn path(&self, span: Span, strs: ~[ast::Ident]) -> ast::Path {
        self.path_all(span, false, strs, None, ~[])
    }
    fn path_ident(&self, span: Span, id: ast::Ident) -> ast::Path {
        self.path(span, ~[id])
    }
    fn path_global(&self, span: Span, strs: ~[ast::Ident]) -> ast::Path {
        self.path_all(span, true, strs, None, ~[])
    }
    fn path_all(&self,
                sp: Span,
                global: bool,
                mut idents: ~[ast::Ident],
                rp: Option<ast::Lifetime>,
                types: ~[ast::Ty])
                -> ast::Path {
        let last_identifier = idents.pop();
        let mut segments: ~[ast::PathSegment] = idents.move_iter()
                                                      .map(|ident| {
            ast::PathSegment {
                identifier: ident,
                lifetime: None,
                types: opt_vec::Empty,
            }
        }).collect();
        segments.push(ast::PathSegment {
            identifier: last_identifier,
            lifetime: rp,
            types: opt_vec::from(types),
        });
        ast::Path {
            span: sp,
            global: global,
            segments: segments,
        }
    }

    fn ty_mt(&self, ty: ast::Ty, mutbl: ast::Mutability) -> ast::mt {
        ast::mt {
            ty: ~ty,
            mutbl: mutbl
        }
    }

    fn ty(&self, span: Span, ty: ast::ty_) -> ast::Ty {
        ast::Ty {
            id: ast::DUMMY_NODE_ID,
            span: span,
            node: ty
        }
    }

    fn ty_path(&self, path: ast::Path, bounds: Option<OptVec<ast::TyParamBound>>)
              -> ast::Ty {
        self.ty(path.span,
                ast::ty_path(path, bounds, ast::DUMMY_NODE_ID))
    }

    // Might need to take bounds as an argument in the future, if you ever want
    // to generate a bounded existential trait type.
    fn ty_ident(&self, span: Span, ident: ast::Ident)
        -> ast::Ty {
        self.ty_path(self.path_ident(span, ident), None)
    }

    fn ty_rptr(&self,
               span: Span,
               ty: ast::Ty,
               lifetime: Option<ast::Lifetime>,
               mutbl: ast::Mutability)
        -> ast::Ty {
        self.ty(span,
                ast::ty_rptr(lifetime, self.ty_mt(ty, mutbl)))
    }
    fn ty_uniq(&self, span: Span, ty: ast::Ty) -> ast::Ty {
        self.ty(span, ast::ty_uniq(self.ty_mt(ty, ast::MutImmutable)))
    }
    fn ty_box(&self, span: Span,
                 ty: ast::Ty, mutbl: ast::Mutability) -> ast::Ty {
        self.ty(span, ast::ty_box(self.ty_mt(ty, mutbl)))
    }

    fn ty_option(&self, ty: ast::Ty) -> ast::Ty {
        self.ty_path(
            self.path_all(dummy_sp(),
                          true,
                          ~[
                              self.ident_of("std"),
                              self.ident_of("option"),
                              self.ident_of("Option")
                          ],
                          None,
                          ~[ ty ]), None)
    }

    fn ty_field_imm(&self, span: Span, name: Ident, ty: ast::Ty) -> ast::TypeField {
        ast::TypeField {
            ident: name,
            mt: ast::mt { ty: ~ty, mutbl: ast::MutImmutable },
            span: span,
        }
    }

    fn ty_infer(&self, span: Span) -> ast::Ty {
        self.ty(span, ast::ty_infer)
    }

    fn ty_nil(&self) -> ast::Ty {
        ast::Ty {
            id: ast::DUMMY_NODE_ID,
            node: ast::ty_nil,
            span: dummy_sp(),
        }
    }

    fn typaram(&self, id: ast::Ident, bounds: OptVec<ast::TyParamBound>) -> ast::TyParam {
        ast::TyParam { ident: id, id: ast::DUMMY_NODE_ID, bounds: bounds }
    }

    // these are strange, and probably shouldn't be used outside of
    // pipes. Specifically, the global version possible generates
    // incorrect code.
    fn ty_vars(&self, ty_params: &OptVec<ast::TyParam>) -> ~[ast::Ty] {
        opt_vec::take_vec(
            ty_params.map(|p| self.ty_ident(dummy_sp(), p.ident)))
    }

    fn ty_vars_global(&self, ty_params: &OptVec<ast::TyParam>) -> ~[ast::Ty] {
        opt_vec::take_vec(
            ty_params.map(|p| self.ty_path(
                self.path_global(dummy_sp(), ~[p.ident]), None)))
    }

    fn strip_bounds(&self, generics: &Generics) -> Generics {
        let new_params = do generics.ty_params.map |ty_param| {
            ast::TyParam { bounds: opt_vec::Empty, ..*ty_param }
        };
        Generics {
            ty_params: new_params,
            .. (*generics).clone()
        }
    }

    fn trait_ref(&self, path: ast::Path) -> ast::trait_ref {
        ast::trait_ref {
            path: path,
            ref_id: ast::DUMMY_NODE_ID
        }
    }

    fn typarambound(&self, path: ast::Path) -> ast::TyParamBound {
        ast::TraitTyParamBound(self.trait_ref(path))
    }

    fn lifetime(&self, span: Span, ident: ast::Ident) -> ast::Lifetime {
        ast::Lifetime { id: ast::DUMMY_NODE_ID, span: span, ident: ident }
    }

    fn stmt_expr(&self, expr: @ast::Expr) -> @ast::Stmt {
        @respan(expr.span, ast::StmtSemi(expr, ast::DUMMY_NODE_ID))
    }

    fn stmt_let(&self, sp: Span, mutbl: bool, ident: ast::Ident, ex: @ast::Expr) -> @ast::Stmt {
        let pat = self.pat_ident(sp, ident);
        let local = @ast::Local {
            is_mutbl: mutbl,
            ty: self.ty_infer(sp),
            pat: pat,
            init: Some(ex),
            id: ast::DUMMY_NODE_ID,
            span: sp,
        };
        let decl = respan(sp, ast::DeclLocal(local));
        @respan(sp, ast::StmtDecl(@decl, ast::DUMMY_NODE_ID))
    }

    fn stmt_let_typed(&self,
                      sp: Span,
                      mutbl: bool,
                      ident: ast::Ident,
                      typ: ast::Ty,
                      ex: @ast::Expr)
                      -> @ast::Stmt {
        let pat = self.pat_ident(sp, ident);
        let local = @ast::Local {
            is_mutbl: mutbl,
            ty: typ,
            pat: pat,
            init: Some(ex),
            id: ast::DUMMY_NODE_ID,
            span: sp,
        };
        let decl = respan(sp, ast::DeclLocal(local));
        @respan(sp, ast::StmtDecl(@decl, ast::DUMMY_NODE_ID))
    }

    fn block(&self, span: Span, stmts: ~[@ast::Stmt], expr: Option<@Expr>) -> ast::Block {
        self.block_all(span, ~[], stmts, expr)
    }

    fn block_expr(&self, expr: @ast::Expr) -> ast::Block {
        self.block_all(expr.span, ~[], ~[], Some(expr))
    }
    fn block_all(&self,
                 span: Span,
                 view_items: ~[ast::view_item],
                 stmts: ~[@ast::Stmt],
                 expr: Option<@ast::Expr>) -> ast::Block {
           ast::Block {
               view_items: view_items,
               stmts: stmts,
               expr: expr,
               id: ast::DUMMY_NODE_ID,
               rules: ast::DefaultBlock,
               span: span,
           }
    }

    fn expr(&self, span: Span, node: ast::Expr_) -> @ast::Expr {
        @ast::Expr {
            id: ast::DUMMY_NODE_ID,
            node: node,
            span: span,
        }
    }

    fn expr_path(&self, path: ast::Path) -> @ast::Expr {
        self.expr(path.span, ast::ExprPath(path))
    }

    fn expr_ident(&self, span: Span, id: ast::Ident) -> @ast::Expr {
        self.expr_path(self.path_ident(span, id))
    }
    fn expr_self(&self, span: Span) -> @ast::Expr {
        self.expr(span, ast::ExprSelf)
    }

    fn expr_binary(&self, sp: Span, op: ast::BinOp,
                   lhs: @ast::Expr, rhs: @ast::Expr) -> @ast::Expr {
        self.expr(sp, ast::ExprBinary(ast::DUMMY_NODE_ID, op, lhs, rhs))
    }

    fn expr_deref(&self, sp: Span, e: @ast::Expr) -> @ast::Expr {
        self.expr_unary(sp, ast::UnDeref, e)
    }
    fn expr_unary(&self, sp: Span, op: ast::UnOp, e: @ast::Expr)
        -> @ast::Expr {
        self.expr(sp, ast::ExprUnary(ast::DUMMY_NODE_ID, op, e))
    }

    fn expr_managed(&self, sp: Span, e: @ast::Expr) -> @ast::Expr {
        self.expr_unary(sp, ast::UnBox(ast::MutImmutable), e)
    }

    fn expr_field_access(&self, sp: Span, expr: @ast::Expr, ident: ast::Ident) -> @ast::Expr {
        self.expr(sp, ast::ExprField(expr, ident, ~[]))
    }
    fn expr_addr_of(&self, sp: Span, e: @ast::Expr) -> @ast::Expr {
        self.expr(sp, ast::ExprAddrOf(ast::MutImmutable, e))
    }
    fn expr_mut_addr_of(&self, sp: Span, e: @ast::Expr) -> @ast::Expr {
        self.expr(sp, ast::ExprAddrOf(ast::MutMutable, e))
    }

    fn expr_call(&self, span: Span, expr: @ast::Expr, args: ~[@ast::Expr]) -> @ast::Expr {
        self.expr(span, ast::ExprCall(expr, args, ast::NoSugar))
    }
    fn expr_call_ident(&self, span: Span, id: ast::Ident, args: ~[@ast::Expr]) -> @ast::Expr {
        self.expr(span,
                  ast::ExprCall(self.expr_ident(span, id), args, ast::NoSugar))
    }
    fn expr_call_global(&self, sp: Span, fn_path: ~[ast::Ident],
                      args: ~[@ast::Expr]) -> @ast::Expr {
        let pathexpr = self.expr_path(self.path_global(sp, fn_path));
        self.expr_call(sp, pathexpr, args)
    }
    fn expr_method_call(&self, span: Span,
                        expr: @ast::Expr,
                        ident: ast::Ident,
                        args: ~[@ast::Expr]) -> @ast::Expr {
        self.expr(span,
                  ast::ExprMethodCall(ast::DUMMY_NODE_ID, expr, ident, ~[], args, ast::NoSugar))
    }
    fn expr_block(&self, b: ast::Block) -> @ast::Expr {
        self.expr(b.span, ast::ExprBlock(b))
    }
    fn field_imm(&self, span: Span, name: Ident, e: @ast::Expr) -> ast::Field {
        ast::Field { ident: name, expr: e, span: span }
    }
    fn expr_struct(&self, span: Span, path: ast::Path, fields: ~[ast::Field]) -> @ast::Expr {
        self.expr(span, ast::ExprStruct(path, fields, None))
    }
    fn expr_struct_ident(&self, span: Span,
                         id: ast::Ident, fields: ~[ast::Field]) -> @ast::Expr {
        self.expr_struct(span, self.path_ident(span, id), fields)
    }

    fn expr_lit(&self, sp: Span, lit: ast::lit_) -> @ast::Expr {
        self.expr(sp, ast::ExprLit(@respan(sp, lit)))
    }
    fn expr_uint(&self, span: Span, i: uint) -> @ast::Expr {
        self.expr_lit(span, ast::lit_uint(i as u64, ast::ty_u))
    }
    fn expr_int(&self, sp: Span, i: int) -> @ast::Expr {
        self.expr_lit(sp, ast::lit_int(i as i64, ast::ty_i))
    }
    fn expr_u8(&self, sp: Span, u: u8) -> @ast::Expr {
        self.expr_lit(sp, ast::lit_uint(u as u64, ast::ty_u8))
    }
    fn expr_bool(&self, sp: Span, value: bool) -> @ast::Expr {
        self.expr_lit(sp, ast::lit_bool(value))
    }

    fn expr_vstore(&self, sp: Span, expr: @ast::Expr, vst: ast::ExprVstore) -> @ast::Expr {
        self.expr(sp, ast::ExprVstore(expr, vst))
    }
    fn expr_vec(&self, sp: Span, exprs: ~[@ast::Expr]) -> @ast::Expr {
        self.expr(sp, ast::ExprVec(exprs, ast::MutImmutable))
    }
    fn expr_vec_uniq(&self, sp: Span, exprs: ~[@ast::Expr]) -> @ast::Expr {
        self.expr_vstore(sp, self.expr_vec(sp, exprs), ast::ExprVstoreUniq)
    }
    fn expr_vec_slice(&self, sp: Span, exprs: ~[@ast::Expr]) -> @ast::Expr {
        self.expr_vstore(sp, self.expr_vec(sp, exprs), ast::ExprVstoreSlice)
    }
    fn expr_str(&self, sp: Span, s: @str) -> @ast::Expr {
        self.expr_lit(sp, ast::lit_str(s))
    }
    fn expr_str_uniq(&self, sp: Span, s: @str) -> @ast::Expr {
        self.expr_vstore(sp, self.expr_str(sp, s), ast::ExprVstoreUniq)
    }


    fn expr_unreachable(&self, span: Span) -> @ast::Expr {
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
                self.expr_str(span, @"internal error: entered unreachable code"),
                self.expr_str(span, loc.file.name),
                self.expr_uint(span, loc.line),
            ])
    }


    fn pat(&self, span: Span, pat: ast::Pat_) -> @ast::Pat {
        @ast::Pat { id: ast::DUMMY_NODE_ID, node: pat, span: span }
    }
    fn pat_wild(&self, span: Span) -> @ast::Pat {
        self.pat(span, ast::PatWild)
    }
    fn pat_lit(&self, span: Span, expr: @ast::Expr) -> @ast::Pat {
        self.pat(span, ast::PatLit(expr))
    }
    fn pat_ident(&self, span: Span, ident: ast::Ident) -> @ast::Pat {
        self.pat_ident_binding_mode(span, ident, ast::BindInfer)
    }

    fn pat_ident_binding_mode(&self,
                              span: Span,
                              ident: ast::Ident,
                              bm: ast::BindingMode) -> @ast::Pat {
        let path = self.path_ident(span, ident);
        let pat = ast::PatIdent(bm, path, None);
        self.pat(span, pat)
    }
    fn pat_enum(&self, span: Span, path: ast::Path, subpats: ~[@ast::Pat]) -> @ast::Pat {
        let pat = ast::PatEnum(path, Some(subpats));
        self.pat(span, pat)
    }
    fn pat_struct(&self, span: Span,
                  path: ast::Path, field_pats: ~[ast::FieldPat]) -> @ast::Pat {
        let pat = ast::PatStruct(path, field_pats, false);
        self.pat(span, pat)
    }

    fn arm(&self, _span: Span, pats: ~[@ast::Pat], expr: @ast::Expr) -> ast::Arm {
        ast::Arm {
            pats: pats,
            guard: None,
            body: self.block_expr(expr)
        }
    }

    fn arm_unreachable(&self, span: Span) -> ast::Arm {
        self.arm(span, ~[self.pat_wild(span)], self.expr_unreachable(span))
    }

    fn expr_match(&self, span: Span, arg: @ast::Expr, arms: ~[ast::Arm]) -> @Expr {
        self.expr(span, ast::ExprMatch(arg, arms))
    }

    fn expr_if(&self, span: Span,
               cond: @ast::Expr, then: @ast::Expr, els: Option<@ast::Expr>) -> @ast::Expr {
        let els = els.map_move(|x| self.expr_block(self.block_expr(x)));
        self.expr(span, ast::ExprIf(cond, self.block_expr(then), els))
    }

    fn lambda_fn_decl(&self, span: Span, fn_decl: ast::fn_decl, blk: ast::Block) -> @ast::Expr {
        self.expr(span, ast::ExprFnBlock(fn_decl, blk))
    }
    fn lambda(&self, span: Span, ids: ~[ast::Ident], blk: ast::Block) -> @ast::Expr {
        let fn_decl = self.fn_decl(
            ids.map(|id| self.arg(span, *id, self.ty_infer(span))),
            self.ty_infer(span));

        self.expr(span, ast::ExprFnBlock(fn_decl, blk))
    }
    #[cfg(stage0)]
    fn lambda0(&self, _span: Span, blk: ast::Block) -> @ast::Expr {
        let ext_cx = *self;
        let blk_e = self.expr(blk.span, ast::ExprBlock(blk.clone()));
        quote_expr!(|| $blk_e )
    }
    #[cfg(not(stage0))]
    fn lambda0(&self, _span: Span, blk: ast::Block) -> @ast::Expr {
        let blk_e = self.expr(blk.span, ast::ExprBlock(blk.clone()));
        quote_expr!(*self, || $blk_e )
    }

    #[cfg(stage0)]
    fn lambda1(&self, _span: Span, blk: ast::Block, ident: ast::Ident) -> @ast::Expr {
        let ext_cx = *self;
        let blk_e = self.expr(blk.span, ast::ExprBlock(blk.clone()));
        quote_expr!(|$ident| $blk_e )
    }
    #[cfg(not(stage0))]
    fn lambda1(&self, _span: Span, blk: ast::Block, ident: ast::Ident) -> @ast::Expr {
        let blk_e = self.expr(blk.span, ast::ExprBlock(blk.clone()));
        quote_expr!(*self, |$ident| $blk_e )
    }

    fn lambda_expr(&self, span: Span, ids: ~[ast::Ident], expr: @ast::Expr) -> @ast::Expr {
        self.lambda(span, ids, self.block_expr(expr))
    }
    fn lambda_expr_0(&self, span: Span, expr: @ast::Expr) -> @ast::Expr {
        self.lambda0(span, self.block_expr(expr))
    }
    fn lambda_expr_1(&self, span: Span, expr: @ast::Expr, ident: ast::Ident) -> @ast::Expr {
        self.lambda1(span, self.block_expr(expr), ident)
    }

    fn lambda_stmts(&self, span: Span, ids: ~[ast::Ident], stmts: ~[@ast::Stmt]) -> @ast::Expr {
        self.lambda(span, ids, self.block(span, stmts, None))
    }
    fn lambda_stmts_0(&self, span: Span, stmts: ~[@ast::Stmt]) -> @ast::Expr {
        self.lambda0(span, self.block(span, stmts, None))
    }
    fn lambda_stmts_1(&self, span: Span, stmts: ~[@ast::Stmt], ident: ast::Ident) -> @ast::Expr {
        self.lambda1(span, self.block(span, stmts, None), ident)
    }

    fn arg(&self, span: Span, ident: ast::Ident, ty: ast::Ty) -> ast::arg {
        let arg_pat = self.pat_ident(span, ident);
        ast::arg {
            is_mutbl: false,
            ty: ty,
            pat: arg_pat,
            id: ast::DUMMY_NODE_ID
        }
    }

    // XXX unused self
    fn fn_decl(&self, inputs: ~[ast::arg], output: ast::Ty) -> ast::fn_decl {
        ast::fn_decl {
            inputs: inputs,
            output: output,
            cf: ast::return_val,
        }
    }

    fn item(&self, span: Span,
            name: Ident, attrs: ~[ast::Attribute], node: ast::item_) -> @ast::item {
        // XXX: Would be nice if our generated code didn't violate
        // Rust coding conventions
        @ast::item { ident: name,
                    attrs: attrs,
                    id: ast::DUMMY_NODE_ID,
                    node: node,
                    vis: ast::public,
                    span: span }
    }

    fn item_fn_poly(&self,
                    span: Span,
                    name: Ident,
                    inputs: ~[ast::arg],
                    output: ast::Ty,
                    generics: Generics,
                    body: ast::Block) -> @ast::item {
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
               span: Span,
               name: Ident,
               inputs: ~[ast::arg],
               output: ast::Ty,
               body: ast::Block
              ) -> @ast::item {
        self.item_fn_poly(
            span,
            name,
            inputs,
            output,
            ast_util::empty_generics(),
            body)
    }

    fn variant(&self, span: Span, name: Ident, tys: ~[ast::Ty]) -> ast::variant {
        let args = tys.move_iter().map(|ty| {
            ast::variant_arg { ty: ty, id: ast::DUMMY_NODE_ID }
        }).collect();

        respan(span,
               ast::variant_ {
                   name: name,
                   attrs: ~[],
                   kind: ast::tuple_variant_kind(args),
                   id: ast::DUMMY_NODE_ID,
                   disr_expr: None,
                   vis: ast::public
               })
    }

    fn item_enum_poly(&self, span: Span, name: Ident,
                      enum_definition: ast::enum_def,
                      generics: Generics) -> @ast::item {
        self.item(span, name, ~[], ast::item_enum(enum_definition, generics))
    }

    fn item_enum(&self, span: Span, name: Ident,
                 enum_definition: ast::enum_def) -> @ast::item {
        self.item_enum_poly(span, name, enum_definition,
                            ast_util::empty_generics())
    }

    fn item_struct(
        &self,
        span: Span,
        name: Ident,
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
        span: Span,
        name: Ident,
        struct_def: ast::struct_def,
        generics: Generics
    ) -> @ast::item {
        self.item(span, name, ~[], ast::item_struct(@struct_def, generics))
    }

    fn item_mod(&self, span: Span, name: Ident,
                attrs: ~[ast::Attribute],
                vi: ~[ast::view_item],
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

    fn item_ty_poly(&self, span: Span, name: Ident, ty: ast::Ty,
                    generics: Generics) -> @ast::item {
        self.item(span, name, ~[], ast::item_ty(ty, generics))
    }

    fn item_ty(&self, span: Span, name: Ident, ty: ast::Ty) -> @ast::item {
        self.item_ty_poly(span, name, ty, ast_util::empty_generics())
    }

    fn attribute(&self, sp: Span, mi: @ast::MetaItem) -> ast::Attribute {
        respan(sp, ast::Attribute_ {
            style: ast::AttrOuter,
            value: mi,
            is_sugared_doc: false,
        })
    }

    fn meta_word(&self, sp: Span, w: @str) -> @ast::MetaItem {
        @respan(sp, ast::MetaWord(w))
    }
    fn meta_list(&self, sp: Span, name: @str, mis: ~[@ast::MetaItem]) -> @ast::MetaItem {
        @respan(sp, ast::MetaList(name, mis))
    }
    fn meta_name_value(&self, sp: Span, name: @str, value: ast::lit_) -> @ast::MetaItem {
        @respan(sp, ast::MetaNameValue(name, respan(sp, value)))
    }

    fn view_use(&self, sp: Span,
                vis: ast::visibility, vp: ~[@ast::view_path]) -> ast::view_item {
        ast::view_item {
            node: ast::view_item_use(vp),
            attrs: ~[],
            vis: vis,
            span: sp
        }
    }

    fn view_use_list(&self, sp: Span, vis: ast::visibility,
                     path: ~[ast::Ident], imports: &[ast::Ident]) -> ast::view_item {
        let imports = do imports.map |id| {
            respan(sp, ast::path_list_ident_ { name: *id, id: ast::DUMMY_NODE_ID })
        };

        self.view_use(sp, vis,
                      ~[@respan(sp,
                                ast::view_path_list(self.path(sp, path),
                                                    imports,
                                                    ast::DUMMY_NODE_ID))])
    }

    fn view_use_glob(&self, sp: Span,
                     vis: ast::visibility, path: ~[ast::Ident]) -> ast::view_item {
        self.view_use(sp, vis,
                      ~[@respan(sp,
                                ast::view_path_glob(self.path(sp, path), ast::DUMMY_NODE_ID))])
    }
}
