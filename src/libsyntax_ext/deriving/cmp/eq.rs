// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use deriving::generic::*;
use deriving::generic::ty::*;

use syntax::ast::{self, Expr, MetaItem};
use syntax::ext::base::{Annotatable, ExtCtxt};
use syntax::ext::build::AstBuilder;
use syntax::ptr::P;
use syntax::symbol::Symbol;
use syntax_pos::Span;

pub fn expand_deriving_eq(cx: &mut ExtCtxt,
                          span: Span,
                          mitem: &MetaItem,
                          item: &Annotatable,
                          push: &mut FnMut(Annotatable)) {
    let inline = cx.meta_word(span, Symbol::intern("inline"));
    let hidden = cx.meta_list_item_word(span, Symbol::intern("hidden"));
    let doc = cx.meta_list(span, Symbol::intern("doc"), vec![hidden]);
    let attrs = vec![cx.attribute(span, inline), cx.attribute(span, doc)];
    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: path_std!(cx, core::cmp::Eq),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        is_unsafe: false,
        supports_unions: true,
        methods: vec![MethodDef {
                          name: "assert_receiver_is_total_eq",
                          generics: LifetimeBounds::empty(),
                          explicit_self: borrowed_explicit_self(),
                          args: vec![],
                          ret_ty: nil_ty(),
                          attributes: attrs,
                          is_unsafe: false,
                          unify_fieldless_variants: true,
                          combine_substructure: combine_substructure(Box::new(|a, b, c| {
                              cs_total_eq_assert(a, b, c)
                          })),
                      }],
        associated_types: Vec::new(),
    };
    trait_def.expand_ext(cx, mitem, item, push, true)
}

fn cs_total_eq_assert(cx: &mut ExtCtxt, trait_span: Span, substr: &Substructure) -> P<Expr> {
    fn assert_ty_bounds(cx: &mut ExtCtxt, stmts: &mut Vec<ast::Stmt>,
                        ty: P<ast::Ty>, span: Span, helper_name: &str) {
        // Generate statement `let _: helper_name<ty>;`,
        // set the expn ID so we can use the unstable struct.
        let span = super::allow_unstable(cx, span, "derive(Eq)");
        let assert_path = cx.path_all(span, true,
                                        cx.std_path(&["cmp", helper_name]),
                                        vec![], vec![ty], vec![]);
        stmts.push(cx.stmt_let_type_only(span, cx.ty_path(assert_path)));
    }
    fn process_variant(cx: &mut ExtCtxt, stmts: &mut Vec<ast::Stmt>, variant: &ast::VariantData) {
        for field in variant.fields() {
            // let _: AssertParamIsEq<FieldTy>;
            assert_ty_bounds(cx, stmts, field.ty.clone(), field.span, "AssertParamIsEq");
        }
    }

    let mut stmts = Vec::new();
    match *substr.fields {
        StaticStruct(vdata, ..) => {
            process_variant(cx, &mut stmts, vdata);
        }
        StaticEnum(enum_def, ..) => {
            for variant in &enum_def.variants {
                process_variant(cx, &mut stmts, &variant.node.data);
            }
        }
        _ => cx.span_bug(trait_span, "unexpected substructure in `derive(Eq)`")
    }
    cx.expr_block(cx.block(trait_span, stmts))
}
