// Functions for building ASTs, without having to fuss with spans.
//
// To start with, it will be use dummy spans, but it might someday do
// something smarter.

use ast::{ident, node_id};
use ast_util::respan;
use codemap::span;
use ext::base::mk_ctxt;

// Transitional reexports so qquote can find the paths it is looking for
mod syntax {
    #[legacy_exports];
    pub use ext;
    pub use parse;
}

fn path(ids: ~[ident], span: span) -> @ast::path {
    @{span: span,
      global: false,
      idents: ids,
      rp: None,
      types: ~[]}
}

fn empty_span() -> span {
    {lo: 0, hi: 0, expn_info: None}
}

trait append_types {
    fn add_ty(ty: @ast::Ty) -> @ast::path;
    fn add_tys(+tys: ~[@ast::Ty]) -> @ast::path;
}

impl @ast::path: append_types {
    fn add_ty(ty: @ast::Ty) -> @ast::path {
        @{types: vec::append_one(self.types, ty),
          .. *self}
    }

    fn add_tys(+tys: ~[@ast::Ty]) -> @ast::path {
        @{types: vec::append(self.types, tys),
          .. *self}
    }
}

trait ext_ctxt_ast_builder {
    fn ty_param(id: ast::ident, +bounds: ~[ast::ty_param_bound])
        -> ast::ty_param;
    fn arg(name: ident, ty: @ast::Ty) -> ast::arg;
    fn expr_block(e: @ast::expr) -> ast::blk;
    fn fn_decl(+inputs: ~[ast::arg], output: @ast::Ty) -> ast::fn_decl;
    fn item(name: ident, span: span, +node: ast::item_) -> @ast::item;
    fn item_fn_poly(name: ident,
                    +inputs: ~[ast::arg],
                    output: @ast::Ty,
                    +ty_params: ~[ast::ty_param],
                    +body: ast::blk) -> @ast::item;
    fn item_fn(name: ident,
               +inputs: ~[ast::arg],
               output: @ast::Ty,
               +body: ast::blk) -> @ast::item;
    fn item_enum_poly(name: ident,
                      span: span,
                      +enum_definition: ast::enum_def,
                      +ty_params: ~[ast::ty_param]) -> @ast::item;
    fn item_enum(name: ident, span: span,
                 +enum_definition: ast::enum_def) -> @ast::item;
    fn variant(name: ident, span: span, +tys: ~[@ast::Ty]) -> ast::variant;
    fn item_mod(name: ident, span: span, +items: ~[@ast::item]) -> @ast::item;
    fn ty_path_ast_builder(path: @ast::path) -> @ast::Ty;
    fn item_ty_poly(name: ident,
                    span: span,
                    ty: @ast::Ty,
                    +params: ~[ast::ty_param]) -> @ast::item;
    fn item_ty(name: ident, span: span, ty: @ast::Ty) -> @ast::item;
    fn ty_vars(+ty_params: ~[ast::ty_param]) -> ~[@ast::Ty];
    fn ty_field_imm(name: ident, ty: @ast::Ty) -> ast::ty_field;
    fn ty_rec(+v: ~[ast::ty_field]) -> @ast::Ty;
    fn field_imm(name: ident, e: @ast::expr) -> ast::field;
    fn rec(+v: ~[ast::field]) -> @ast::expr;
    fn block(+stmts: ~[@ast::stmt], e: @ast::expr) -> ast::blk;
    fn stmt_let(ident: ident, e: @ast::expr) -> @ast::stmt;
    fn stmt_expr(e: @ast::expr) -> @ast::stmt;
    fn block_expr(b: ast::blk) -> @ast::expr;
    fn empty_span() -> span;
    fn ty_option(ty: @ast::Ty) -> @ast::Ty;
}

impl ext_ctxt: ext_ctxt_ast_builder {
    fn ty_option(ty: @ast::Ty) -> @ast::Ty {
        self.ty_path_ast_builder(path(~[self.ident_of(~"Option")],
                                      self.empty_span())
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

    fn move_expr(e: @ast::expr) -> @ast::expr {
        @{id: self.next_id(),
          callee_id: self.next_id(),
          node: ast::expr_unary_move(e),
          span: e.span}
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
                                                 path(~[ident],
                                                      self.empty_span()),
                                                 None),
                            span: self.empty_span()},
                     init: Some(self.move_expr(e)),
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
          node: ast::expr_rec(fields, None),
          span: self.empty_span()}
    }

    fn ty_field_imm(name: ident, ty: @ast::Ty) -> ast::ty_field {
        {node: {ident: name, mt: { ty: ty, mutbl: ast::m_imm } },
          span: self.empty_span()}
    }

    fn ty_rec(+fields: ~[ast::ty_field]) -> @ast::Ty {
        @{id: self.next_id(),
          node: ast::ty_rec(fields),
          span: self.empty_span()}
    }

    fn ty_infer() -> @ast::Ty {
        @{id: self.next_id(),
          node: ast::ty_infer,
          span: self.empty_span()}
    }

    fn ty_param(id: ast::ident, +bounds: ~[ast::ty_param_bound])
        -> ast::ty_param
    {
        {ident: id, id: self.next_id(), bounds: @bounds}
    }

    fn arg(name: ident, ty: @ast::Ty) -> ast::arg {
        {mode: ast::infer(self.next_id()),
         ty: ty,
         ident: name,
         id: self.next_id()}
    }

    fn block(+stmts: ~[@ast::stmt], e: @ast::expr) -> ast::blk {
        let blk = {view_items: ~[],
                   stmts: stmts,
                   expr: Some(e),
                   id: self.next_id(),
                   rules: ast::default_blk};

        {node: blk,
         span: self.empty_span()}
    }

    fn expr_block(e: @ast::expr) -> ast::blk {
        self.block(~[], e)
    }

    fn fn_decl(+inputs: ~[ast::arg],
               output: @ast::Ty) -> ast::fn_decl {
        {inputs: inputs,
         output: output,
         cf: ast::return_val}
    }

    fn item(name: ident,
            span: span,
            +node: ast::item_) -> @ast::item {

        // XXX: Would be nice if our generated code didn't violate
        // Rust coding conventions
        let non_camel_case_attribute = respan(self.empty_span(), {
            style: ast::attr_outer,
            value: respan(self.empty_span(),
                          ast::meta_list(~"allow", ~[
                              @respan(self.empty_span(),
                                      ast::meta_word(~"non_camel_case_types"))
                          ])),
            is_sugared_doc: false
        });

        @{ident: name,
         attrs: ~[non_camel_case_attribute],
         id: self.next_id(),
         node: node,
         vis: ast::public,
         span: span}
    }

    fn item_fn_poly(name: ident,
                    +inputs: ~[ast::arg],
                    output: @ast::Ty,
                    +ty_params: ~[ast::ty_param],
                    +body: ast::blk) -> @ast::item {
        self.item(name,
                  self.empty_span(),
                  ast::item_fn(self.fn_decl(inputs, output),
                               ast::impure_fn,
                               ty_params,
                               body))
    }

    fn item_fn(name: ident,
               +inputs: ~[ast::arg],
               output: @ast::Ty,
               +body: ast::blk) -> @ast::item {
        self.item_fn_poly(name, inputs, output, ~[], body)
    }

    fn item_enum_poly(name: ident,
                      span: span,
                      +enum_definition: ast::enum_def,
                      +ty_params: ~[ast::ty_param]) -> @ast::item {
        self.item(name, span, ast::item_enum(enum_definition, ty_params))
    }

    fn item_enum(name: ident, span: span,
                 +enum_definition: ast::enum_def) -> @ast::item {
        self.item_enum_poly(name, span, enum_definition, ~[])
    }

    fn variant(name: ident,
               span: span,
               +tys: ~[@ast::Ty]) -> ast::variant {
        let args = tys.map(|ty| {ty: *ty, id: self.next_id()});

        {node: {name: name,
                attrs: ~[],
                kind: ast::tuple_variant_kind(args),
                id: self.next_id(),
                disr_expr: None,
                vis: ast::public},
         span: span}
    }

    fn item_mod(name: ident,
                span: span,
                +items: ~[@ast::item]) -> @ast::item {
        self.item(name,
                  span,
                  ast::item_mod({
                      view_items: ~[],
                      items: items}))
    }

    fn ty_path_ast_builder(path: @ast::path) -> @ast::Ty {
        @{id: self.next_id(),
          node: ast::ty_path(path, self.next_id()),
          span: path.span}
    }

    fn ty_nil_ast_builder() -> @ast::Ty {
        @{id: self.next_id(),
          node: ast::ty_nil,
          span: self.empty_span()}
    }

    fn item_ty_poly(name: ident,
                    span: span,
                    ty: @ast::Ty,
                    +params: ~[ast::ty_param]) -> @ast::item {
        self.item(name, span, ast::item_ty(ty, params))
    }

    fn item_ty(name: ident, span: span, ty: @ast::Ty) -> @ast::item {
        self.item_ty_poly(name, span, ty, ~[])
    }

    fn ty_vars(+ty_params: ~[ast::ty_param]) -> ~[@ast::Ty] {
        ty_params.map(|p| self.ty_path_ast_builder(
            path(~[p.ident], self.empty_span())))
    }
}
