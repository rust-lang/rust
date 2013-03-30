// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Functions for building ASTs, without having to fuss with spans.
//
// To start with, it will be use dummy spans, but it might someday do
// something smarter.

use core::prelude::*;

use abi::AbiSet;
use ast::{ident, node_id};
use ast;
use ast_util;
use codemap::{span, respan, dummy_sp, spanned};
use codemap;
use ext::base::ext_ctxt;
use ext::quote::rt::*;
use opt_vec;
use opt_vec::OptVec;

use core::vec;

// Transitional reexports so qquote can find the paths it is looking for
mod syntax {
    pub use ext;
    pub use parse;
}

pub fn path(+ids: ~[ident], span: span) -> @ast::path {
    @ast::path { span: span,
                 global: false,
                 idents: ids,
                 rp: None,
                 types: ~[] }
}

pub fn path_global(+ids: ~[ident], span: span) -> @ast::path {
    @ast::path { span: span,
                 global: true,
                 idents: ids,
                 rp: None,
                 types: ~[] }
}

pub trait append_types {
    fn add_ty(&self, ty: @ast::Ty) -> @ast::path;
    fn add_tys(&self, +tys: ~[@ast::Ty]) -> @ast::path;
}

impl append_types for @ast::path {
    fn add_ty(&self, ty: @ast::Ty) -> @ast::path {
        @ast::path {
            types: vec::append_one(copy self.types, ty),
            .. copy **self
        }
    }

    fn add_tys(&self, +tys: ~[@ast::Ty]) -> @ast::path {
        @ast::path {
            types: vec::append(copy self.types, tys),
            .. copy **self
        }
    }
}

pub trait ext_ctxt_ast_builder {
    fn ty_param(&self, id: ast::ident, bounds: @OptVec<ast::TyParamBound>)
        -> ast::TyParam;
    fn arg(&self, name: ident, ty: @ast::Ty) -> ast::arg;
    fn expr_block(&self, e: @ast::expr) -> ast::blk;
    fn fn_decl(&self, +inputs: ~[ast::arg], output: @ast::Ty) -> ast::fn_decl;
    fn item(&self, name: ident, span: span, +node: ast::item_) -> @ast::item;
    fn item_fn_poly(&self,
                    ame: ident,
                    +inputs: ~[ast::arg],
                    output: @ast::Ty,
                    +generics: Generics,
                    +body: ast::blk) -> @ast::item;
    fn item_fn(&self,
               name: ident,
               +inputs: ~[ast::arg],
               output: @ast::Ty,
               +body: ast::blk) -> @ast::item;
    fn item_enum_poly(&self,
                      name: ident,
                      span: span,
                      +enum_definition: ast::enum_def,
                      +generics: Generics) -> @ast::item;
    fn item_enum(&self,
                 name: ident,
                 span: span,
                 +enum_definition: ast::enum_def) -> @ast::item;
    fn item_struct_poly(&self,
                        name: ident,
                        span: span,
                        +struct_def: ast::struct_def,
                        +generics: Generics) -> @ast::item;
    fn item_struct(&self,
                   name: ident,
                   span: span,
                   +struct_def: ast::struct_def) -> @ast::item;
    fn struct_expr(&self,
                   path: @ast::path,
                   +fields: ~[ast::field]) -> @ast::expr;
    fn variant(&self,
               name: ident,
               span: span,
               +tys: ~[@ast::Ty]) -> ast::variant;
    fn item_mod(&self,
                name: ident,
                span: span,
                +items: ~[@ast::item]) -> @ast::item;
    fn ty_path_ast_builder(&self, path: @ast::path) -> @ast::Ty;
    fn item_ty_poly(&self,
                    name: ident,
                    span: span,
                    ty: @ast::Ty,
                    +generics: Generics) -> @ast::item;
    fn item_ty(&self, name: ident, span: span, ty: @ast::Ty) -> @ast::item;
    fn ty_vars(&self, ty_params: &OptVec<ast::TyParam>) -> ~[@ast::Ty];
    fn ty_vars_global(&self, ty_params: &OptVec<ast::TyParam>) -> ~[@ast::Ty];
    fn ty_field_imm(&self, name: ident, ty: @ast::Ty) -> ast::ty_field;
    fn field_imm(&self, name: ident, e: @ast::expr) -> ast::field;
    fn block(&self, +stmts: ~[@ast::stmt], e: @ast::expr) -> ast::blk;
    fn stmt_let(&self, ident: ident, e: @ast::expr) -> @ast::stmt;
    fn stmt_expr(&self, e: @ast::expr) -> @ast::stmt;
    fn block_expr(&self, +b: ast::blk) -> @ast::expr;
    fn ty_option(&self, ty: @ast::Ty) -> @ast::Ty;
    fn ty_infer(&self) -> @ast::Ty;
    fn ty_nil_ast_builder(&self) -> @ast::Ty;
    fn strip_bounds(&self, bounds: &Generics) -> Generics;
}

impl ext_ctxt_ast_builder for @ext_ctxt {
    fn ty_option(&self, ty: @ast::Ty) -> @ast::Ty {
        self.ty_path_ast_builder(path_global(~[
            self.ident_of(~"core"),
            self.ident_of(~"option"),
            self.ident_of(~"Option")
        ], dummy_sp()).add_ty(ty))
    }

    fn block_expr(&self, +b: ast::blk) -> @ast::expr {
        @expr {
            id: self.next_id(),
            callee_id: self.next_id(),
            node: ast::expr_block(b),
            span: dummy_sp(),
        }
    }

    fn stmt_expr(&self, e: @ast::expr) -> @ast::stmt {
        @spanned { node: ast::stmt_expr(e, self.next_id()),
                   span: dummy_sp()}
    }

    fn stmt_let(&self, ident: ident, e: @ast::expr) -> @ast::stmt {
        let ext_cx = *self;
        quote_stmt!( let $ident = $e; )
    }

    fn field_imm(&self, name: ident, e: @ast::expr) -> ast::field {
        spanned {
            node: ast::field_ { mutbl: ast::m_imm, ident: name, expr: e },
            span: dummy_sp(),
        }
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

    fn arg(&self, name: ident, ty: @ast::Ty) -> ast::arg {
        ast::arg {
            mode: ast::infer(self.next_id()),
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

    fn block(&self, +stmts: ~[@ast::stmt], e: @ast::expr) -> ast::blk {
        let blk = ast::blk_ {
            view_items: ~[],
            stmts: stmts,
            expr: Some(e),
            id: self.next_id(),
            rules: ast::default_blk,
        };

        spanned { node: blk, span: dummy_sp() }
    }

    fn expr_block(&self, e: @ast::expr) -> ast::blk {
        self.block(~[], e)
    }

    fn fn_decl(&self, +inputs: ~[ast::arg],
               output: @ast::Ty) -> ast::fn_decl {
        ast::fn_decl {
            inputs: inputs,
            output: output,
            cf: ast::return_val,
        }
    }

    fn item(&self, name: ident, span: span,
            +node: ast::item_) -> @ast::item {

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
                    +inputs: ~[ast::arg],
                    output: @ast::Ty,
                    +generics: Generics,
                    +body: ast::blk) -> @ast::item {
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
               +inputs: ~[ast::arg],
               output: @ast::Ty,
               +body: ast::blk
    ) -> @ast::item {
        self.item_fn_poly(
            name,
            inputs,
            output,
            ast_util::empty_generics(),
            body
        )
    }

    fn item_enum_poly(&self, name: ident, span: span,
                      +enum_definition: ast::enum_def,
                      +generics: Generics) -> @ast::item {
        self.item(name, span, ast::item_enum(enum_definition, generics))
    }

    fn item_enum(&self, name: ident, span: span,
                 +enum_definition: ast::enum_def) -> @ast::item {
        self.item_enum_poly(name, span, enum_definition,
                            ast_util::empty_generics())
    }

    fn item_struct(
        &self, name: ident,
        span: span,
        +struct_def: ast::struct_def
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
        +struct_def: ast::struct_def,
        +generics: Generics
    ) -> @ast::item {
        self.item(name, span, ast::item_struct(@struct_def, generics))
    }

    fn struct_expr(&self, path: @ast::path,
                   +fields: ~[ast::field]) -> @ast::expr {
        @ast::expr {
            id: self.next_id(),
            callee_id: self.next_id(),
            node: ast::expr_struct(path, fields, None),
            span: dummy_sp()
        }
    }

    fn variant(&self, name: ident, span: span,
               +tys: ~[@ast::Ty]) -> ast::variant {
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

    fn item_mod(&self, name: ident, span: span,
                +items: ~[@ast::item]) -> @ast::item {

        // XXX: Total hack: import `core::kinds::Owned` to work around a
        // parser bug whereby `fn f<T:::kinds::Owned>` doesn't parse.
        let vi = ast::view_item_use(~[
            @codemap::spanned {
                node: ast::view_path_simple(
                    self.ident_of(~"Owned"),
                    path(
                        ~[
                            self.ident_of(~"core"),
                            self.ident_of(~"kinds"),
                            self.ident_of(~"Owned")
                        ],
                        codemap::dummy_sp()
                    ),
                    ast::type_value_ns,
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

    fn ty_path_ast_builder(&self, path: @ast::path) -> @ast::Ty {
        @ast::Ty {
            id: self.next_id(),
            node: ast::ty_path(path, self.next_id()),
            span: path.span,
        }
    }

    fn ty_nil_ast_builder(&self) -> @ast::Ty {
        @ast::Ty {
            id: self.next_id(),
            node: ast::ty_nil,
            span: dummy_sp(),
        }
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

    fn item_ty_poly(&self, name: ident, span: span, ty: @ast::Ty,
                    +generics: Generics) -> @ast::item {
        self.item(name, span, ast::item_ty(ty, generics))
    }

    fn item_ty(&self, name: ident, span: span, ty: @ast::Ty) -> @ast::item {
        self.item_ty_poly(name, span, ty, ast_util::empty_generics())
    }

    fn ty_vars(&self, ty_params: &OptVec<ast::TyParam>) -> ~[@ast::Ty] {
        opt_vec::take_vec(
            ty_params.map(|p| self.ty_path_ast_builder(
                path(~[p.ident], dummy_sp()))))
    }

    fn ty_vars_global(&self,
                      ty_params: &OptVec<ast::TyParam>) -> ~[@ast::Ty] {
        opt_vec::take_vec(
            ty_params.map(|p| self.ty_path_ast_builder(
                path(~[p.ident], dummy_sp()))))
    }
}
