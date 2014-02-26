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
use ast::{P, Ident};
use ast;
use ast_util;
use codemap::{Span, respan, DUMMY_SP};
use ext::base::ExtCtxt;
use ext::quote::rt::*;
use fold::Folder;
use opt_vec;
use opt_vec::OptVec;
use parse::token::special_idents;
use parse::token;

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
                lifetimes: OptVec<ast::Lifetime>,
                types: ~[P<ast::Ty>])
        -> ast::Path;

    // types
    fn ty_mt(&self, ty: P<ast::Ty>, mutbl: ast::Mutability) -> ast::MutTy;

    fn ty(&self, span: Span, ty: ast::Ty_) -> P<ast::Ty>;
    fn ty_path(&self, ast::Path, Option<OptVec<ast::TyParamBound>>) -> P<ast::Ty>;
    fn ty_ident(&self, span: Span, idents: ast::Ident) -> P<ast::Ty>;

    fn ty_rptr(&self, span: Span,
               ty: P<ast::Ty>,
               lifetime: Option<ast::Lifetime>,
               mutbl: ast::Mutability) -> P<ast::Ty>;
    fn ty_uniq(&self, span: Span, ty: P<ast::Ty>) -> P<ast::Ty>;

    fn ty_option(&self, ty: P<ast::Ty>) -> P<ast::Ty>;
    fn ty_infer(&self, sp: Span) -> P<ast::Ty>;
    fn ty_nil(&self) -> P<ast::Ty>;

    fn ty_vars(&self, ty_params: &OptVec<ast::TyParam>) -> ~[P<ast::Ty>];
    fn ty_vars_global(&self, ty_params: &OptVec<ast::TyParam>) -> ~[P<ast::Ty>];
    fn ty_field_imm(&self, span: Span, name: Ident, ty: P<ast::Ty>) -> ast::TypeField;
    fn strip_bounds(&self, bounds: &Generics) -> Generics;

    fn typaram(&self,
               id: ast::Ident,
               bounds: OptVec<ast::TyParamBound>,
               default: Option<P<ast::Ty>>) -> ast::TyParam;

    fn trait_ref(&self, path: ast::Path) -> ast::TraitRef;
    fn typarambound(&self, path: ast::Path) -> ast::TyParamBound;
    fn lifetime(&self, span: Span, ident: ast::Name) -> ast::Lifetime;

    // statements
    fn stmt_expr(&self, expr: @ast::Expr) -> @ast::Stmt;
    fn stmt_let(&self, sp: Span, mutbl: bool, ident: ast::Ident, ex: @ast::Expr) -> @ast::Stmt;
    fn stmt_let_typed(&self,
                      sp: Span,
                      mutbl: bool,
                      ident: ast::Ident,
                      typ: P<ast::Ty>,
                      ex: @ast::Expr)
                      -> @ast::Stmt;

    // blocks
    fn block(&self, span: Span, stmts: ~[@ast::Stmt], expr: Option<@ast::Expr>) -> P<ast::Block>;
    fn block_expr(&self, expr: @ast::Expr) -> P<ast::Block>;
    fn block_all(&self, span: Span,
                 view_items: ~[ast::ViewItem],
                 stmts: ~[@ast::Stmt],
                 expr: Option<@ast::Expr>) -> P<ast::Block>;

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
    fn expr_block(&self, b: P<ast::Block>) -> @ast::Expr;
    fn expr_cast(&self, sp: Span, expr: @ast::Expr, ty: P<ast::Ty>) -> @ast::Expr;

    fn field_imm(&self, span: Span, name: Ident, e: @ast::Expr) -> ast::Field;
    fn expr_struct(&self, span: Span, path: ast::Path, fields: ~[ast::Field]) -> @ast::Expr;
    fn expr_struct_ident(&self, span: Span, id: ast::Ident, fields: ~[ast::Field]) -> @ast::Expr;

    fn expr_lit(&self, sp: Span, lit: ast::Lit_) -> @ast::Expr;

    fn expr_uint(&self, span: Span, i: uint) -> @ast::Expr;
    fn expr_int(&self, sp: Span, i: int) -> @ast::Expr;
    fn expr_u8(&self, sp: Span, u: u8) -> @ast::Expr;
    fn expr_bool(&self, sp: Span, value: bool) -> @ast::Expr;

    fn expr_vstore(&self, sp: Span, expr: @ast::Expr, vst: ast::ExprVstore) -> @ast::Expr;
    fn expr_vec(&self, sp: Span, exprs: ~[@ast::Expr]) -> @ast::Expr;
    fn expr_vec_uniq(&self, sp: Span, exprs: ~[@ast::Expr]) -> @ast::Expr;
    fn expr_vec_slice(&self, sp: Span, exprs: ~[@ast::Expr]) -> @ast::Expr;
    fn expr_str(&self, sp: Span, s: InternedString) -> @ast::Expr;
    fn expr_str_uniq(&self, sp: Span, s: InternedString) -> @ast::Expr;

    fn expr_some(&self, sp: Span, expr: @ast::Expr) -> @ast::Expr;
    fn expr_none(&self, sp: Span) -> @ast::Expr;

    fn expr_fail(&self, span: Span, msg: InternedString) -> @ast::Expr;
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

    fn lambda_fn_decl(&self, span: Span,
                      fn_decl: P<ast::FnDecl>, blk: P<ast::Block>) -> @ast::Expr;

    fn lambda(&self, span: Span, ids: ~[ast::Ident], blk: P<ast::Block>) -> @ast::Expr;
    fn lambda0(&self, span: Span, blk: P<ast::Block>) -> @ast::Expr;
    fn lambda1(&self, span: Span, blk: P<ast::Block>, ident: ast::Ident) -> @ast::Expr;

    fn lambda_expr(&self, span: Span, ids: ~[ast::Ident], blk: @ast::Expr) -> @ast::Expr;
    fn lambda_expr_0(&self, span: Span, expr: @ast::Expr) -> @ast::Expr;
    fn lambda_expr_1(&self, span: Span, expr: @ast::Expr, ident: ast::Ident) -> @ast::Expr;

    fn lambda_stmts(&self, span: Span, ids: ~[ast::Ident], blk: ~[@ast::Stmt]) -> @ast::Expr;
    fn lambda_stmts_0(&self, span: Span, stmts: ~[@ast::Stmt]) -> @ast::Expr;
    fn lambda_stmts_1(&self, span: Span, stmts: ~[@ast::Stmt], ident: ast::Ident) -> @ast::Expr;

    // items
    fn item(&self, span: Span,
            name: Ident, attrs: ~[ast::Attribute], node: ast::Item_) -> @ast::Item;

    fn arg(&self, span: Span, name: Ident, ty: P<ast::Ty>) -> ast::Arg;
    // FIXME unused self
    fn fn_decl(&self, inputs: ~[ast::Arg], output: P<ast::Ty>) -> P<ast::FnDecl>;

    fn item_fn_poly(&self,
                    span: Span,
                    name: Ident,
                    inputs: ~[ast::Arg],
                    output: P<ast::Ty>,
                    generics: Generics,
                    body: P<ast::Block>) -> @ast::Item;
    fn item_fn(&self,
               span: Span,
               name: Ident,
               inputs: ~[ast::Arg],
               output: P<ast::Ty>,
               body: P<ast::Block>) -> @ast::Item;

    fn variant(&self, span: Span, name: Ident, tys: ~[P<ast::Ty>]) -> ast::Variant;
    fn item_enum_poly(&self,
                      span: Span,
                      name: Ident,
                      enum_definition: ast::EnumDef,
                      generics: Generics) -> @ast::Item;
    fn item_enum(&self, span: Span, name: Ident, enum_def: ast::EnumDef) -> @ast::Item;

    fn item_struct_poly(&self,
                        span: Span,
                        name: Ident,
                        struct_def: ast::StructDef,
                        generics: Generics) -> @ast::Item;
    fn item_struct(&self, span: Span, name: Ident, struct_def: ast::StructDef) -> @ast::Item;

    fn item_mod(&self, span: Span,
                name: Ident, attrs: ~[ast::Attribute],
                vi: ~[ast::ViewItem], items: ~[@ast::Item]) -> @ast::Item;

    fn item_ty_poly(&self,
                    span: Span,
                    name: Ident,
                    ty: P<ast::Ty>,
                    generics: Generics) -> @ast::Item;
    fn item_ty(&self, span: Span, name: Ident, ty: P<ast::Ty>) -> @ast::Item;

    fn attribute(&self, sp: Span, mi: @ast::MetaItem) -> ast::Attribute;

    fn meta_word(&self, sp: Span, w: InternedString) -> @ast::MetaItem;
    fn meta_list(&self,
                 sp: Span,
                 name: InternedString,
                 mis: ~[@ast::MetaItem])
                 -> @ast::MetaItem;
    fn meta_name_value(&self,
                       sp: Span,
                       name: InternedString,
                       value: ast::Lit_)
                       -> @ast::MetaItem;

    fn view_use(&self, sp: Span,
                vis: ast::Visibility, vp: ~[@ast::ViewPath]) -> ast::ViewItem;
    fn view_use_simple(&self, sp: Span, vis: ast::Visibility, path: ast::Path) -> ast::ViewItem;
    fn view_use_simple_(&self, sp: Span, vis: ast::Visibility,
                        ident: ast::Ident, path: ast::Path) -> ast::ViewItem;
    fn view_use_list(&self, sp: Span, vis: ast::Visibility,
                     path: ~[ast::Ident], imports: &[ast::Ident]) -> ast::ViewItem;
    fn view_use_glob(&self, sp: Span,
                     vis: ast::Visibility, path: ~[ast::Ident]) -> ast::ViewItem;
}

impl<'a> AstBuilder for ExtCtxt<'a> {
    fn path(&self, span: Span, strs: ~[ast::Ident]) -> ast::Path {
        self.path_all(span, false, strs, opt_vec::Empty, ~[])
    }
    fn path_ident(&self, span: Span, id: ast::Ident) -> ast::Path {
        self.path(span, ~[id])
    }
    fn path_global(&self, span: Span, strs: ~[ast::Ident]) -> ast::Path {
        self.path_all(span, true, strs, opt_vec::Empty, ~[])
    }
    fn path_all(&self,
                sp: Span,
                global: bool,
                mut idents: ~[ast::Ident],
                lifetimes: OptVec<ast::Lifetime>,
                types: ~[P<ast::Ty>])
                -> ast::Path {
        let last_identifier = idents.pop().unwrap();
        let mut segments: ~[ast::PathSegment] = idents.move_iter()
                                                      .map(|ident| {
            ast::PathSegment {
                identifier: ident,
                lifetimes: opt_vec::Empty,
                types: opt_vec::Empty,
            }
        }).collect();
        segments.push(ast::PathSegment {
            identifier: last_identifier,
            lifetimes: lifetimes,
            types: opt_vec::from(types),
        });
        ast::Path {
            span: sp,
            global: global,
            segments: segments,
        }
    }

    fn ty_mt(&self, ty: P<ast::Ty>, mutbl: ast::Mutability) -> ast::MutTy {
        ast::MutTy {
            ty: ty,
            mutbl: mutbl
        }
    }

    fn ty(&self, span: Span, ty: ast::Ty_) -> P<ast::Ty> {
        P(ast::Ty {
            id: ast::DUMMY_NODE_ID,
            span: span,
            node: ty
        })
    }

    fn ty_path(&self, path: ast::Path, bounds: Option<OptVec<ast::TyParamBound>>)
              -> P<ast::Ty> {
        self.ty(path.span,
                ast::TyPath(path, bounds, ast::DUMMY_NODE_ID))
    }

    // Might need to take bounds as an argument in the future, if you ever want
    // to generate a bounded existential trait type.
    fn ty_ident(&self, span: Span, ident: ast::Ident)
        -> P<ast::Ty> {
        self.ty_path(self.path_ident(span, ident), None)
    }

    fn ty_rptr(&self,
               span: Span,
               ty: P<ast::Ty>,
               lifetime: Option<ast::Lifetime>,
               mutbl: ast::Mutability)
        -> P<ast::Ty> {
        self.ty(span,
                ast::TyRptr(lifetime, self.ty_mt(ty, mutbl)))
    }

    fn ty_uniq(&self, span: Span, ty: P<ast::Ty>) -> P<ast::Ty> {
        self.ty(span, ast::TyUniq(ty))
    }

    fn ty_option(&self, ty: P<ast::Ty>) -> P<ast::Ty> {
        self.ty_path(
            self.path_all(DUMMY_SP,
                          true,
                          ~[
                              self.ident_of("std"),
                              self.ident_of("option"),
                              self.ident_of("Option")
                          ],
                          opt_vec::Empty,
                          ~[ ty ]), None)
    }

    fn ty_field_imm(&self, span: Span, name: Ident, ty: P<ast::Ty>) -> ast::TypeField {
        ast::TypeField {
            ident: name,
            mt: ast::MutTy { ty: ty, mutbl: ast::MutImmutable },
            span: span,
        }
    }

    fn ty_infer(&self, span: Span) -> P<ast::Ty> {
        self.ty(span, ast::TyInfer)
    }

    fn ty_nil(&self) -> P<ast::Ty> {
        P(ast::Ty {
            id: ast::DUMMY_NODE_ID,
            node: ast::TyNil,
            span: DUMMY_SP,
        })
    }

    fn typaram(&self,
               id: ast::Ident,
               bounds: OptVec<ast::TyParamBound>,
               default: Option<P<ast::Ty>>) -> ast::TyParam {
        ast::TyParam {
            ident: id,
            id: ast::DUMMY_NODE_ID,
            bounds: bounds,
            default: default
        }
    }

    // these are strange, and probably shouldn't be used outside of
    // pipes. Specifically, the global version possible generates
    // incorrect code.
    fn ty_vars(&self, ty_params: &OptVec<ast::TyParam>) -> ~[P<ast::Ty>] {
        opt_vec::take_vec(
            ty_params.map(|p| self.ty_ident(DUMMY_SP, p.ident)))
    }

    fn ty_vars_global(&self, ty_params: &OptVec<ast::TyParam>) -> ~[P<ast::Ty>] {
        opt_vec::take_vec(
            ty_params.map(|p| self.ty_path(
                self.path_global(DUMMY_SP, ~[p.ident]), None)))
    }

    fn strip_bounds(&self, generics: &Generics) -> Generics {
        let new_params = generics.ty_params.map(|ty_param| {
            ast::TyParam { bounds: opt_vec::Empty, ..*ty_param }
        });
        Generics {
            ty_params: new_params,
            .. (*generics).clone()
        }
    }

    fn trait_ref(&self, path: ast::Path) -> ast::TraitRef {
        ast::TraitRef {
            path: path,
            ref_id: ast::DUMMY_NODE_ID
        }
    }

    fn typarambound(&self, path: ast::Path) -> ast::TyParamBound {
        ast::TraitTyParamBound(self.trait_ref(path))
    }

    fn lifetime(&self, span: Span, ident: ast::Name) -> ast::Lifetime {
        ast::Lifetime { id: ast::DUMMY_NODE_ID, span: span, ident: ident }
    }

    fn stmt_expr(&self, expr: @ast::Expr) -> @ast::Stmt {
        @respan(expr.span, ast::StmtSemi(expr, ast::DUMMY_NODE_ID))
    }

    fn stmt_let(&self, sp: Span, mutbl: bool, ident: ast::Ident, ex: @ast::Expr) -> @ast::Stmt {
        let pat = if mutbl {
            self.pat_ident_binding_mode(sp, ident, ast::BindByValue(ast::MutMutable))
        } else {
            self.pat_ident(sp, ident)
        };
        let local = @ast::Local {
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
                      typ: P<ast::Ty>,
                      ex: @ast::Expr)
                      -> @ast::Stmt {
        let pat = if mutbl {
            self.pat_ident_binding_mode(sp, ident, ast::BindByValue(ast::MutMutable))
        } else {
            self.pat_ident(sp, ident)
        };
        let local = @ast::Local {
            ty: typ,
            pat: pat,
            init: Some(ex),
            id: ast::DUMMY_NODE_ID,
            span: sp,
        };
        let decl = respan(sp, ast::DeclLocal(local));
        @respan(sp, ast::StmtDecl(@decl, ast::DUMMY_NODE_ID))
    }

    fn block(&self, span: Span, stmts: ~[@ast::Stmt], expr: Option<@Expr>) -> P<ast::Block> {
        self.block_all(span, ~[], stmts, expr)
    }

    fn block_expr(&self, expr: @ast::Expr) -> P<ast::Block> {
        self.block_all(expr.span, ~[], ~[], Some(expr))
    }
    fn block_all(&self,
                 span: Span,
                 view_items: ~[ast::ViewItem],
                 stmts: ~[@ast::Stmt],
                 expr: Option<@ast::Expr>) -> P<ast::Block> {
            P(ast::Block {
               view_items: view_items,
               stmts: stmts,
               expr: expr,
               id: ast::DUMMY_NODE_ID,
               rules: ast::DefaultBlock,
               span: span,
            })
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
        self.expr_ident(span, special_idents::self_)
    }

    fn expr_binary(&self, sp: Span, op: ast::BinOp,
                   lhs: @ast::Expr, rhs: @ast::Expr) -> @ast::Expr {
        self.expr(sp, ast::ExprBinary(op, lhs, rhs))
    }

    fn expr_deref(&self, sp: Span, e: @ast::Expr) -> @ast::Expr {
        self.expr_unary(sp, ast::UnDeref, e)
    }
    fn expr_unary(&self, sp: Span, op: ast::UnOp, e: @ast::Expr) -> @ast::Expr {
        self.expr(sp, ast::ExprUnary(op, e))
    }

    fn expr_managed(&self, sp: Span, e: @ast::Expr) -> @ast::Expr {
        self.expr_unary(sp, ast::UnBox, e)
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
        self.expr(span, ast::ExprCall(expr, args))
    }
    fn expr_call_ident(&self, span: Span, id: ast::Ident, args: ~[@ast::Expr]) -> @ast::Expr {
        self.expr(span, ast::ExprCall(self.expr_ident(span, id), args))
    }
    fn expr_call_global(&self, sp: Span, fn_path: ~[ast::Ident],
                      args: ~[@ast::Expr]) -> @ast::Expr {
        let pathexpr = self.expr_path(self.path_global(sp, fn_path));
        self.expr_call(sp, pathexpr, args)
    }
    fn expr_method_call(&self, span: Span,
                        expr: @ast::Expr,
                        ident: ast::Ident,
                        mut args: ~[@ast::Expr]) -> @ast::Expr {
        args.unshift(expr);
        self.expr(span, ast::ExprMethodCall(ident, ~[], args))
    }
    fn expr_block(&self, b: P<ast::Block>) -> @ast::Expr {
        self.expr(b.span, ast::ExprBlock(b))
    }
    fn field_imm(&self, span: Span, name: Ident, e: @ast::Expr) -> ast::Field {
        ast::Field { ident: respan(span, name), expr: e, span: span }
    }
    fn expr_struct(&self, span: Span, path: ast::Path, fields: ~[ast::Field]) -> @ast::Expr {
        self.expr(span, ast::ExprStruct(path, fields, None))
    }
    fn expr_struct_ident(&self, span: Span,
                         id: ast::Ident, fields: ~[ast::Field]) -> @ast::Expr {
        self.expr_struct(span, self.path_ident(span, id), fields)
    }

    fn expr_lit(&self, sp: Span, lit: ast::Lit_) -> @ast::Expr {
        self.expr(sp, ast::ExprLit(@respan(sp, lit)))
    }
    fn expr_uint(&self, span: Span, i: uint) -> @ast::Expr {
        self.expr_lit(span, ast::LitUint(i as u64, ast::TyU))
    }
    fn expr_int(&self, sp: Span, i: int) -> @ast::Expr {
        self.expr_lit(sp, ast::LitInt(i as i64, ast::TyI))
    }
    fn expr_u8(&self, sp: Span, u: u8) -> @ast::Expr {
        self.expr_lit(sp, ast::LitUint(u as u64, ast::TyU8))
    }
    fn expr_bool(&self, sp: Span, value: bool) -> @ast::Expr {
        self.expr_lit(sp, ast::LitBool(value))
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
    fn expr_str(&self, sp: Span, s: InternedString) -> @ast::Expr {
        self.expr_lit(sp, ast::LitStr(s, ast::CookedStr))
    }
    fn expr_str_uniq(&self, sp: Span, s: InternedString) -> @ast::Expr {
        self.expr_vstore(sp, self.expr_str(sp, s), ast::ExprVstoreUniq)
    }


    fn expr_cast(&self, sp: Span, expr: @ast::Expr, ty: P<ast::Ty>) -> @ast::Expr {
        self.expr(sp, ast::ExprCast(expr, ty))
    }


    fn expr_some(&self, sp: Span, expr: @ast::Expr) -> @ast::Expr {
        let some = ~[
            self.ident_of("std"),
            self.ident_of("option"),
            self.ident_of("Some"),
        ];
        self.expr_call_global(sp, some, ~[expr])
    }

    fn expr_none(&self, sp: Span) -> @ast::Expr {
        let none = self.path_global(sp, ~[
            self.ident_of("std"),
            self.ident_of("option"),
            self.ident_of("None"),
        ]);
        self.expr_path(none)
    }

    fn expr_fail(&self, span: Span, msg: InternedString) -> @ast::Expr {
        let loc = self.codemap().lookup_char_pos(span.lo);
        self.expr_call_global(
            span,
            ~[
                self.ident_of("std"),
                self.ident_of("rt"),
                self.ident_of("begin_unwind"),
            ],
            ~[
                self.expr_str(span, msg),
                self.expr_str(span,
                              token::intern_and_get_ident(loc.file.name)),
                self.expr_uint(span, loc.line),
            ])
    }

    fn expr_unreachable(&self, span: Span) -> @ast::Expr {
        self.expr_fail(span,
                       InternedString::new(
                           "internal error: entered unreachable code"))
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
        self.pat_ident_binding_mode(span, ident, ast::BindByValue(ast::MutImmutable))
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
        let els = els.map(|x| self.expr_block(self.block_expr(x)));
        self.expr(span, ast::ExprIf(cond, self.block_expr(then), els))
    }

    fn lambda_fn_decl(&self, span: Span,
                      fn_decl: P<ast::FnDecl>, blk: P<ast::Block>) -> @ast::Expr {
        self.expr(span, ast::ExprFnBlock(fn_decl, blk))
    }
    fn lambda(&self, span: Span, ids: ~[ast::Ident], blk: P<ast::Block>) -> @ast::Expr {
        let fn_decl = self.fn_decl(
            ids.map(|id| self.arg(span, *id, self.ty_infer(span))),
            self.ty_infer(span));

        self.expr(span, ast::ExprFnBlock(fn_decl, blk))
    }
    fn lambda0(&self, _span: Span, blk: P<ast::Block>) -> @ast::Expr {
        let blk_e = self.expr(blk.span, ast::ExprBlock(blk));
        quote_expr!(self, || $blk_e )
    }

    fn lambda1(&self, _span: Span, blk: P<ast::Block>, ident: ast::Ident) -> @ast::Expr {
        let blk_e = self.expr(blk.span, ast::ExprBlock(blk));
        quote_expr!(self, |$ident| $blk_e )
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

    fn arg(&self, span: Span, ident: ast::Ident, ty: P<ast::Ty>) -> ast::Arg {
        let arg_pat = self.pat_ident(span, ident);
        ast::Arg {
            ty: ty,
            pat: arg_pat,
            id: ast::DUMMY_NODE_ID
        }
    }

    // FIXME unused self
    fn fn_decl(&self, inputs: ~[ast::Arg], output: P<ast::Ty>) -> P<ast::FnDecl> {
        P(ast::FnDecl {
            inputs: inputs,
            output: output,
            cf: ast::Return,
            variadic: false
        })
    }

    fn item(&self, span: Span,
            name: Ident, attrs: ~[ast::Attribute], node: ast::Item_) -> @ast::Item {
        // FIXME: Would be nice if our generated code didn't violate
        // Rust coding conventions
        @ast::Item { ident: name,
                    attrs: attrs,
                    id: ast::DUMMY_NODE_ID,
                    node: node,
                    vis: ast::Inherited,
                    span: span }
    }

    fn item_fn_poly(&self,
                    span: Span,
                    name: Ident,
                    inputs: ~[ast::Arg],
                    output: P<ast::Ty>,
                    generics: Generics,
                    body: P<ast::Block>) -> @ast::Item {
        self.item(span,
                  name,
                  ~[],
                  ast::ItemFn(self.fn_decl(inputs, output),
                              ast::ImpureFn,
                              AbiSet::Rust(),
                              generics,
                              body))
    }

    fn item_fn(&self,
               span: Span,
               name: Ident,
               inputs: ~[ast::Arg],
               output: P<ast::Ty>,
               body: P<ast::Block>
              ) -> @ast::Item {
        self.item_fn_poly(
            span,
            name,
            inputs,
            output,
            ast_util::empty_generics(),
            body)
    }

    fn variant(&self, span: Span, name: Ident, tys: ~[P<ast::Ty>]) -> ast::Variant {
        let args = tys.move_iter().map(|ty| {
            ast::VariantArg { ty: ty, id: ast::DUMMY_NODE_ID }
        }).collect();

        respan(span,
               ast::Variant_ {
                   name: name,
                   attrs: ~[],
                   kind: ast::TupleVariantKind(args),
                   id: ast::DUMMY_NODE_ID,
                   disr_expr: None,
                   vis: ast::Public
               })
    }

    fn item_enum_poly(&self, span: Span, name: Ident,
                      enum_definition: ast::EnumDef,
                      generics: Generics) -> @ast::Item {
        self.item(span, name, ~[], ast::ItemEnum(enum_definition, generics))
    }

    fn item_enum(&self, span: Span, name: Ident,
                 enum_definition: ast::EnumDef) -> @ast::Item {
        self.item_enum_poly(span, name, enum_definition,
                            ast_util::empty_generics())
    }

    fn item_struct(&self, span: Span, name: Ident,
                   struct_def: ast::StructDef) -> @ast::Item {
        self.item_struct_poly(
            span,
            name,
            struct_def,
            ast_util::empty_generics()
        )
    }

    fn item_struct_poly(&self, span: Span, name: Ident,
        struct_def: ast::StructDef, generics: Generics) -> @ast::Item {
        self.item(span, name, ~[], ast::ItemStruct(@struct_def, generics))
    }

    fn item_mod(&self, span: Span, name: Ident,
                attrs: ~[ast::Attribute],
                vi: ~[ast::ViewItem],
                items: ~[@ast::Item]) -> @ast::Item {
        self.item(
            span,
            name,
            attrs,
            ast::ItemMod(ast::Mod {
                view_items: vi,
                items: items,
            })
        )
    }

    fn item_ty_poly(&self, span: Span, name: Ident, ty: P<ast::Ty>,
                    generics: Generics) -> @ast::Item {
        self.item(span, name, ~[], ast::ItemTy(ty, generics))
    }

    fn item_ty(&self, span: Span, name: Ident, ty: P<ast::Ty>) -> @ast::Item {
        self.item_ty_poly(span, name, ty, ast_util::empty_generics())
    }

    fn attribute(&self, sp: Span, mi: @ast::MetaItem) -> ast::Attribute {
        respan(sp, ast::Attribute_ {
            style: ast::AttrOuter,
            value: mi,
            is_sugared_doc: false,
        })
    }

    fn meta_word(&self, sp: Span, w: InternedString) -> @ast::MetaItem {
        @respan(sp, ast::MetaWord(w))
    }
    fn meta_list(&self,
                 sp: Span,
                 name: InternedString,
                 mis: ~[@ast::MetaItem])
                 -> @ast::MetaItem {
        @respan(sp, ast::MetaList(name, mis))
    }
    fn meta_name_value(&self,
                       sp: Span,
                       name: InternedString,
                       value: ast::Lit_)
                       -> @ast::MetaItem {
        @respan(sp, ast::MetaNameValue(name, respan(sp, value)))
    }

    fn view_use(&self, sp: Span,
                vis: ast::Visibility, vp: ~[@ast::ViewPath]) -> ast::ViewItem {
        ast::ViewItem {
            node: ast::ViewItemUse(vp),
            attrs: ~[],
            vis: vis,
            span: sp
        }
    }

    fn view_use_simple(&self, sp: Span, vis: ast::Visibility, path: ast::Path) -> ast::ViewItem {
        let last = path.segments.last().unwrap().identifier;
        self.view_use_simple_(sp, vis, last, path)
    }

    fn view_use_simple_(&self, sp: Span, vis: ast::Visibility,
                        ident: ast::Ident, path: ast::Path) -> ast::ViewItem {
        self.view_use(sp, vis,
                      ~[@respan(sp,
                                ast::ViewPathSimple(ident,
                                                    path,
                                                    ast::DUMMY_NODE_ID))])
    }

    fn view_use_list(&self, sp: Span, vis: ast::Visibility,
                     path: ~[ast::Ident], imports: &[ast::Ident]) -> ast::ViewItem {
        let imports = imports.map(|id| {
            respan(sp, ast::PathListIdent_ { name: *id, id: ast::DUMMY_NODE_ID })
        });

        self.view_use(sp, vis,
                      ~[@respan(sp,
                                ast::ViewPathList(self.path(sp, path),
                                                  imports,
                                                  ast::DUMMY_NODE_ID))])
    }

    fn view_use_glob(&self, sp: Span,
                     vis: ast::Visibility, path: ~[ast::Ident]) -> ast::ViewItem {
        self.view_use(sp, vis,
                      ~[@respan(sp,
                                ast::ViewPathGlob(self.path(sp, path), ast::DUMMY_NODE_ID))])
    }
}

struct Duplicator<'a> {
    cx: &'a ExtCtxt<'a>,
}

impl<'a> Folder for Duplicator<'a> {
    fn new_id(&mut self, _: NodeId) -> NodeId {
        ast::DUMMY_NODE_ID
    }
}

pub trait Duplicate {
    //
    // Duplication functions
    //
    // These functions just duplicate AST nodes.
    //

    fn duplicate(&self, cx: &ExtCtxt) -> Self;
}

impl Duplicate for @ast::Expr {
    fn duplicate(&self, cx: &ExtCtxt) -> @ast::Expr {
        let mut folder = Duplicator {
            cx: cx,
        };
        folder.fold_expr(*self)
    }
}
