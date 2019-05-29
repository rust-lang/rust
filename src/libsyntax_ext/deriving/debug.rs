use crate::deriving::path_std;
use crate::deriving::generic::*;
use crate::deriving::generic::ty::*;

use rustc_data_structures::thin_vec::ThinVec;

use syntax::ast::{self, Ident};
use syntax::ast::{Expr, MetaItem};
use syntax::ext::base::{Annotatable, ExtCtxt};
use syntax::ext::build::AstBuilder;
use syntax::ptr::P;
use syntax::symbol::sym;
use syntax_pos::{DUMMY_SP, Span};

pub fn expand_deriving_debug(cx: &mut ExtCtxt<'_>,
                             span: Span,
                             mitem: &MetaItem,
                             item: &Annotatable,
                             push: &mut dyn FnMut(Annotatable)) {
    // &mut ::std::fmt::Formatter
    let fmtr = Ptr(Box::new(Literal(path_std!(cx, fmt::Formatter))),
                   Borrowed(None, ast::Mutability::Mutable));

    let trait_def = TraitDef {
        span,
        attributes: Vec::new(),
        path: path_std!(cx, fmt::Debug),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        is_unsafe: false,
        supports_unions: false,
        methods: vec![MethodDef {
                          name: "fmt",
                          generics: LifetimeBounds::empty(),
                          explicit_self: borrowed_explicit_self(),
                          args: vec![(fmtr, "f")],
                          ret_ty: Literal(path_std!(cx, fmt::Result)),
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
fn show_substructure(cx: &mut ExtCtxt<'_>, span: Span, substr: &Substructure<'_>) -> P<Expr> {
    // build fmt.debug_struct(<name>).field(<fieldname>, &<fieldval>)....build()
    // or fmt.debug_tuple(<name>).field(&<fieldval>)....build()
    // based on the "shape".
    let (ident, vdata, fields) = match substr.fields {
        Struct(vdata, fields) => (substr.type_ident, *vdata, fields),
        EnumMatching(_, _, v, fields) => (v.node.ident, &v.node.data, fields),
        EnumNonMatchingCollapsed(..) |
        StaticStruct(..) |
        StaticEnum(..) => cx.span_bug(span, "nonsensical .fields in `#[derive(Debug)]`"),
    };

    // We want to make sure we have the ctxt set so that we can use unstable methods
    let span = span.with_ctxt(cx.backtrace());
    let name = cx.expr_lit(span, ast::LitKind::Str(ident.name, ast::StrStyle::Cooked));
    let builder = Ident::from_str("debug_trait_builder").gensym();
    let builder_expr = cx.expr_ident(span, builder.clone());

    let fmt = substr.nonself_args[0].clone();

    let mut stmts = vec![];
    match vdata {
        ast::VariantData::Tuple(..) | ast::VariantData::Unit(..) => {
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
                                                Ident::with_empty_ctxt(sym::field),
                                                vec![field]);

                // Use `let _ = expr;` to avoid triggering the
                // unused_results lint.
                stmts.push(stmt_let_undescore(cx, span, expr));
            }
        }
        ast::VariantData::Struct(..) => {
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
                                                Ident::with_empty_ctxt(sym::field),
                                                vec![name, field]);
                stmts.push(stmt_let_undescore(cx, span, expr));
            }
        }
    }

    let expr = cx.expr_method_call(span, builder_expr, Ident::from_str("finish"), vec![]);

    stmts.push(cx.stmt_expr(expr));
    let block = cx.block(span, stmts);
    cx.expr_block(block)
}

fn stmt_let_undescore(cx: &mut ExtCtxt<'_>, sp: Span, expr: P<ast::Expr>) -> ast::Stmt {
    let local = P(ast::Local {
        pat: cx.pat_wild(sp),
        ty: None,
        init: Some(expr),
        id: ast::DUMMY_NODE_ID,
        span: sp,
        attrs: ThinVec::new(),
    });
    ast::Stmt {
        id: ast::DUMMY_NODE_ID,
        node: ast::StmtKind::Local(local),
        span: sp,
    }
}
