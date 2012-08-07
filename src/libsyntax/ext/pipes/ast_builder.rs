// Functions for building ASTs, without having to fuss with spans.
//
// To start with, it will be use dummy spans, but it might someday do
// something smarter.

import ast::{ident, node_id};
import codemap::span;
import ext::base::mk_ctxt;

// Transitional reexports so qquote can find the paths it is looking for
mod syntax {
    import ext;
    export ext;
    import parse;
    export parse;
}

fn ident(s: &str) -> ast::ident {
    @(s.to_unique())
}

fn path(id: ident, span: span) -> @ast::path {
    @{span: span,
      global: false,
      idents: ~[id],
      rp: none,
      types: ~[]}
}

fn empty_span() -> span {
    {lo: 0, hi: 0, expn_info: none}
}

trait append_types {
    fn add_ty(ty: @ast::ty) -> @ast::path;
    fn add_tys(+tys: ~[@ast::ty]) -> @ast::path;
}

impl methods of append_types for @ast::path {
    fn add_ty(ty: @ast::ty) -> @ast::path {
        @{types: vec::append_one(self.types, ty)
          with *self}
    }

    fn add_tys(+tys: ~[@ast::ty]) -> @ast::path {
        @{types: vec::append(self.types, tys)
          with *self}
    }
}

trait ext_ctxt_ast_builder {
    fn ty_param(id: ast::ident, +bounds: ~[ast::ty_param_bound])
        -> ast::ty_param;
    fn arg(name: ident, ty: @ast::ty) -> ast::arg;
    fn arg_mode(name: ident, ty: @ast::ty, mode: ast::rmode) -> ast::arg;
    fn expr_block(e: @ast::expr) -> ast::blk;
    fn fn_decl(+inputs: ~[ast::arg], output: @ast::ty) -> ast::fn_decl;
    fn item(name: ident, +node: ast::item_) -> @ast::item;
    fn item_fn_poly(name: ident,
                    +inputs: ~[ast::arg],
                    output: @ast::ty,
                    +ty_params: ~[ast::ty_param],
                    +body: ast::blk) -> @ast::item;
    fn item_fn(name: ident,
               +inputs: ~[ast::arg],
               output: @ast::ty,
               +body: ast::blk) -> @ast::item;
    fn item_enum_poly(name: ident,
                      +variants: ~[ast::variant],
                      +ty_params: ~[ast::ty_param]) -> @ast::item;
    fn item_enum(name: ident, +variants: ~[ast::variant]) -> @ast::item;
    fn variant(name: ident, +tys: ~[@ast::ty]) -> ast::variant;
    fn item_mod(name: ident, +items: ~[@ast::item]) -> @ast::item;
    fn ty_path_ast_builder(path: @ast::path) -> @ast::ty;
    fn item_ty_poly(name: ident,
                    ty: @ast::ty,
                    +params: ~[ast::ty_param]) -> @ast::item;
    fn item_ty(name: ident, ty: @ast::ty) -> @ast::item;
    fn ty_vars(+ty_params: ~[ast::ty_param]) -> ~[@ast::ty];
    fn ty_field_imm(name: ident, ty: @ast::ty) -> ast::ty_field;
    fn ty_rec(+~[ast::ty_field]) -> @ast::ty;
    fn field_imm(name: ident, e: @ast::expr) -> ast::field;
    fn rec(+~[ast::field]) -> @ast::expr;
    fn block(+stmts: ~[@ast::stmt], e: @ast::expr) -> ast::blk;
    fn stmt_let(ident: ident, e: @ast::expr) -> @ast::stmt;
    fn stmt_expr(e: @ast::expr) -> @ast::stmt;
    fn block_expr(b: ast::blk) -> @ast::expr;
    fn empty_span() -> span;
    fn ty_option(ty: @ast::ty) -> @ast::ty;
}

impl ast_builder of ext_ctxt_ast_builder for ext_ctxt {
    fn ty_option(ty: @ast::ty) -> @ast::ty {
        self.ty_path_ast_builder(path(@~"option", self.empty_span())
                                 .add_ty(ty))
    }

    fn empty_span() -> span {
        {lo: 0, hi: 0, expn_info: self.backtrace()}
    }

    fn block_expr(b: ast::blk) -> @ast::expr {
        @{id: self.next_id(),
          callee_id: self.next_id(),
          node: ast::expr_block(b),
          span: self.empty_span()}
    }

    fn stmt_expr(e: @ast::expr) -> @ast::stmt {
        @{node: ast::stmt_expr(e, self.next_id()),
          span: self.empty_span()}
    }

    fn stmt_let(ident: ident, e: @ast::expr) -> @ast::stmt {
        // If the quasiquoter could interpolate idents, this is all
        // we'd need.
        //
        //let ext_cx = self;
        //#ast[stmt] { let $(ident) = $(e) }

        @{node: ast::stmt_decl(@{node: ast::decl_local(~[
            @{node: {is_mutbl: false,
                     ty: self.ty_infer(),
                     pat: @{id: self.next_id(),
                            node: ast::pat_ident(ast::bind_by_implicit_ref,
                                                 path(ident,
                                                      self.empty_span()),
                                                 none),
                            span: self.empty_span()},
                     init: some({op: ast::init_move,
                                 expr: e}),
                     id: self.next_id()},
              span: self.empty_span()}]),
                               span: self.empty_span()}, self.next_id()),
         span: self.empty_span()}
    }

    fn field_imm(name: ident, e: @ast::expr) -> ast::field {
        {node: {mutbl: ast::m_imm, ident: name, expr: e},
         span: self.empty_span()}
    }

    fn rec(+fields: ~[ast::field]) -> @ast::expr {
        @{id: self.next_id(),
          callee_id: self.next_id(),
          node: ast::expr_rec(fields, none),
          span: self.empty_span()}
    }

    fn ty_field_imm(name: ident, ty: @ast::ty) -> ast::ty_field {
        {node: {ident: name, mt: { ty: ty, mutbl: ast::m_imm } },
          span: self.empty_span()}
    }

    fn ty_rec(+fields: ~[ast::ty_field]) -> @ast::ty {
        @{id: self.next_id(),
          node: ast::ty_rec(fields),
          span: self.empty_span()}
    }

    fn ty_infer() -> @ast::ty {
        @{id: self.next_id(),
          node: ast::ty_infer,
          span: self.empty_span()}
    }

    fn ty_param(id: ast::ident, +bounds: ~[ast::ty_param_bound])
        -> ast::ty_param
    {
        {ident: id, id: self.next_id(), bounds: @bounds}
    }

    fn arg(name: ident, ty: @ast::ty) -> ast::arg {
        {mode: ast::infer(self.next_id()),
         ty: ty,
         ident: name,
         id: self.next_id()}
    }

    fn arg_mode(name: ident, ty: @ast::ty, mode: ast::rmode) -> ast::arg {
        {mode: ast::expl(mode),
         ty: ty,
         ident: name,
         id: self.next_id()}
    }

    fn block(+stmts: ~[@ast::stmt], e: @ast::expr) -> ast::blk {
        let blk = {view_items: ~[],
                   stmts: stmts,
                   expr: some(e),
                   id: self.next_id(),
                   rules: ast::default_blk};

        {node: blk,
         span: self.empty_span()}
    }

    fn expr_block(e: @ast::expr) -> ast::blk {
        self.block(~[], e)
    }

    fn fn_decl(+inputs: ~[ast::arg],
               output: @ast::ty) -> ast::fn_decl {
        {inputs: inputs,
         output: output,
         purity: ast::impure_fn,
         cf: ast::return_val}
    }

    fn item(name: ident,
            +node: ast::item_) -> @ast::item {
        @{ident: name,
         attrs: ~[],
         id: self.next_id(),
         node: node,
         vis: ast::public,
         span: self.empty_span()}
    }

    fn item_fn_poly(name: ident,
                    +inputs: ~[ast::arg],
                    output: @ast::ty,
                    +ty_params: ~[ast::ty_param],
                    +body: ast::blk) -> @ast::item {
        self.item(name,
                  ast::item_fn(self.fn_decl(inputs, output),
                               ty_params,
                               body))
    }

    fn item_fn(name: ident,
               +inputs: ~[ast::arg],
               output: @ast::ty,
               +body: ast::blk) -> @ast::item {
        self.item_fn_poly(name, inputs, output, ~[], body)
    }

    fn item_enum_poly(name: ident,
                      +variants: ~[ast::variant],
                      +ty_params: ~[ast::ty_param]) -> @ast::item {
        self.item(name,
                  ast::item_enum(variants,
                                 ty_params))
    }

    fn item_enum(name: ident,
                 +variants: ~[ast::variant]) -> @ast::item {
        self.item_enum_poly(name, variants, ~[])
    }

    fn variant(name: ident,
               +tys: ~[@ast::ty]) -> ast::variant {
        let args = tys.map(|ty| {ty: ty, id: self.next_id()});

        {node: {name: name,
                attrs: ~[],
                args: args,
                id: self.next_id(),
                disr_expr: none,
                vis: ast::public},
         span: self.empty_span()}
    }

    fn item_mod(name: ident,
                +items: ~[@ast::item]) -> @ast::item {
        self.item(name,
                  ast::item_mod({
                      view_items: ~[],
                      items: items}))
    }

    fn ty_path_ast_builder(path: @ast::path) -> @ast::ty {
        @{id: self.next_id(),
          node: ast::ty_path(path, self.next_id()),
          span: self.empty_span()}
    }

    fn ty_nil_ast_builder() -> @ast::ty {
        @{id: self.next_id(),
          node: ast::ty_nil,
          span: self.empty_span()}
    }

    fn item_ty_poly(name: ident,
                    ty: @ast::ty,
                    +params: ~[ast::ty_param]) -> @ast::item {
        self.item(name, ast::item_ty(ty, params))
    }

    fn item_ty(name: ident, ty: @ast::ty) -> @ast::item {
        self.item_ty_poly(name, ty, ~[])
    }

    fn ty_vars(+ty_params: ~[ast::ty_param]) -> ~[@ast::ty] {
        ty_params.map(|p| self.ty_path_ast_builder(
            path(p.ident, self.empty_span())))
    }
}
