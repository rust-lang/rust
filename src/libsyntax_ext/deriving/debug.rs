// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
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

use syntax::ast::{self, Ident};
use syntax::ast::{Expr, MetaItem};
use syntax::ext::base::{Annotatable, ExtCtxt};
use syntax::ext::build::AstBuilder;
use syntax::ptr::P;
use syntax_pos::{DUMMY_SP, Span};

pub fn expand_deriving_debug(cx: &mut ExtCtxt,
                             span: Span,
                             mitem: &MetaItem,
                             item: &Annotatable,
                             push: &mut FnMut(Annotatable)) {
    // &mut ::std::fmt::Formatter
    let fmtr = Ptr(Box::new(Literal(path_std!(cx, core::fmt::Formatter))),
                   Borrowed(None, ast::Mutability::Mutable));

    let trait_def = TraitDef {
        span: span,
        attributes: Vec::new(),
        path: path_std!(cx, core::fmt::Debug),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        is_unsafe: false,
        supports_unions: false,
        methods: vec![MethodDef {
                          name: "fmt",
                          generics: LifetimeBounds::empty(),
                          explicit_self: borrowed_explicit_self(),
                          args: vec![fmtr],
                          ret_ty: Literal(path_std!(cx, core::fmt::Result)),
                          attributes: Vec::new(),
                          is_unsafe: false,
                          unify_fieldless_variants: false,
                          combine_substructure: combine_substructure(Box::new(|a, b, c| {
                              show_substructure(a, b, c)
                          })),
                      }],
        associated_types: Vec::new(),
    };
    trait_def.expand(cx, mitem, item, push)
}

/// We use the debug builders to do the heavy lifting here
fn show_substructure(cx: &mut ExtCtxt, span: Span, substr: &Substructure) -> P<Expr> {
    // build fmt.debug_struct(<name>).field(<fieldname>, &<fieldval>)....build()
    // or fmt.debug_tuple(<name>).field(&<fieldval>)....build()
    // based on the "shape".
    let (ident, is_struct) = match *substr.fields {
        Struct(vdata, _) => (substr.type_ident, vdata.is_struct()),
        EnumMatching(_, v, _) => (v.node.name, v.node.data.is_struct()),
        EnumNonMatchingCollapsed(..) |
        StaticStruct(..) |
        StaticEnum(..) => cx.span_bug(span, "nonsensical .fields in `#[derive(Debug)]`"),
    };

    // We want to make sure we have the expn_id set so that we can use unstable methods
    let span = Span { expn_id: cx.backtrace(), ..span };
    let name = cx.expr_lit(span, ast::LitKind::Str(ident.name, ast::StrStyle::Cooked));
    let builder = Ident::from_str("builder");
    let builder_expr = cx.expr_ident(span, builder.clone());

    let fmt = substr.nonself_args[0].clone();

    let mut stmts = match *substr.fields {
        Struct(_, ref fields) |
        EnumMatching(.., ref fields) => {
            let mut stmts = vec![];
            if !is_struct {
                // tuple struct/"normal" variant
                let expr =
                    cx.expr_method_call(span, fmt, Ident::from_str("debug_tuple"), vec![name]);
                stmts.push(cx.stmt_let(DUMMY_SP, true, builder, expr));

                for field in fields {
                    // Use double indirection to make sure this works for unsized types
                    let field = cx.expr_addr_of(field.span, field.self_.clone());
                    let field = cx.expr_addr_of(field.span, field);

                    let expr = cx.expr_method_call(span,
                                                   builder_expr.clone(),
                                                   Ident::from_str("field"),
                                                   vec![field]);

                    // Use `let _ = expr;` to avoid triggering the
                    // unused_results lint.
                    stmts.push(stmt_let_undescore(cx, span, expr));
                }
            } else {
                // normal struct/struct variant
                let expr =
                    cx.expr_method_call(span, fmt, Ident::from_str("debug_struct"), vec![name]);
                stmts.push(cx.stmt_let(DUMMY_SP, true, builder, expr));

                for field in fields {
                    let name = cx.expr_lit(field.span,
                                           ast::LitKind::Str(field.name.unwrap().name,
                                                             ast::StrStyle::Cooked));

                    // Use double indirection to make sure this works for unsized types
                    let field = cx.expr_addr_of(field.span, field.self_.clone());
                    let field = cx.expr_addr_of(field.span, field);
                    let expr = cx.expr_method_call(span,
                                                   builder_expr.clone(),
                                                   Ident::from_str("field"),
                                                   vec![name, field]);
                    stmts.push(stmt_let_undescore(cx, span, expr));
                }
            }
            stmts
        }
        _ => unreachable!(),
    };

    let expr = cx.expr_method_call(span, builder_expr, Ident::from_str("finish"), vec![]);

    stmts.push(cx.stmt_expr(expr));
    let block = cx.block(span, stmts);
    cx.expr_block(block)
}

fn stmt_let_undescore(cx: &mut ExtCtxt, sp: Span, expr: P<ast::Expr>) -> ast::Stmt {
    let local = P(ast::Local {
        pat: cx.pat_wild(sp),
        ty: None,
        init: Some(expr),
        id: ast::DUMMY_NODE_ID,
        span: sp,
        attrs: ast::ThinVec::new(),
    });
    ast::Stmt {
        id: ast::DUMMY_NODE_ID,
        node: ast::StmtKind::Local(local),
        span: sp,
    }
}
