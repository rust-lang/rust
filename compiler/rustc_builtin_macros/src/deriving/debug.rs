use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::path_std;

use rustc_ast::ptr::P;
use rustc_ast::{self as ast, Expr, MetaItem};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{sym, Ident};
use rustc_span::{Span, DUMMY_SP};

pub fn expand_deriving_debug(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
) {
    // &mut ::std::fmt::Formatter
    let fmtr =
        Ptr(Box::new(Literal(path_std!(fmt::Formatter))), Borrowed(None, ast::Mutability::Mut));

    let trait_def = TraitDef {
        span,
        attributes: Vec::new(),
        path: path_std!(fmt::Debug),
        additional_bounds: Vec::new(),
        generics: Bounds::empty(),
        is_unsafe: false,
        supports_unions: false,
        methods: vec![MethodDef {
            name: sym::fmt,
            generics: Bounds::empty(),
            explicit_self: borrowed_explicit_self(),
            args: vec![(fmtr, sym::f)],
            ret_ty: Literal(path_std!(fmt::Result)),
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
        EnumMatching(_, _, v, fields) => (v.ident, &v.data, fields),
        EnumNonMatchingCollapsed(..) | StaticStruct(..) | StaticEnum(..) => {
            cx.span_bug(span, "nonsensical .fields in `#[derive(Debug)]`")
        }
    };

    // We want to make sure we have the ctxt set so that we can use unstable methods
    let span = cx.with_def_site_ctxt(span);
    let name = cx.expr_lit(span, ast::LitKind::Str(ident.name, ast::StrStyle::Cooked));
    let builder = Ident::new(sym::debug_trait_builder, span);
    let builder_expr = cx.expr_ident(span, builder);

    let fmt = substr.nonself_args[0].clone();

    let mut stmts = Vec::with_capacity(fields.len() + 2);
    match vdata {
        ast::VariantData::Tuple(..) | ast::VariantData::Unit(..) => {
            // tuple struct/"normal" variant
            let expr =
                cx.expr_method_call(span, fmt, Ident::new(sym::debug_tuple, span), vec![name]);
            stmts.push(cx.stmt_let(span, true, builder, expr));

            for field in fields {
                // Use double indirection to make sure this works for unsized types
                let field = cx.expr_addr_of(field.span, field.self_.clone());
                let field = cx.expr_addr_of(field.span, field);

                let expr = cx.expr_method_call(
                    span,
                    builder_expr.clone(),
                    Ident::new(sym::field, span),
                    vec![field],
                );

                // Use `let _ = expr;` to avoid triggering the
                // unused_results lint.
                stmts.push(stmt_let_underscore(cx, span, expr));
            }
        }
        ast::VariantData::Struct(..) => {
            // normal struct/struct variant
            let expr =
                cx.expr_method_call(span, fmt, Ident::new(sym::debug_struct, span), vec![name]);
            stmts.push(cx.stmt_let(DUMMY_SP, true, builder, expr));

            for field in fields {
                let name = cx.expr_lit(
                    field.span,
                    ast::LitKind::Str(field.name.unwrap().name, ast::StrStyle::Cooked),
                );

                // Use double indirection to make sure this works for unsized types
                let field = cx.expr_addr_of(field.span, field.self_.clone());
                let field = cx.expr_addr_of(field.span, field);
                let expr = cx.expr_method_call(
                    span,
                    builder_expr.clone(),
                    Ident::new(sym::field, span),
                    vec![name, field],
                );
                stmts.push(stmt_let_underscore(cx, span, expr));
            }
        }
    }

    let expr = cx.expr_method_call(span, builder_expr, Ident::new(sym::finish, span), vec![]);

    stmts.push(cx.stmt_expr(expr));
    let block = cx.block(span, stmts);
    cx.expr_block(block)
}

fn stmt_let_underscore(cx: &mut ExtCtxt<'_>, sp: Span, expr: P<ast::Expr>) -> ast::Stmt {
    let local = P(ast::Local {
        pat: cx.pat_wild(sp),
        ty: None,
        init: Some(expr),
        id: ast::DUMMY_NODE_ID,
        span: sp,
        attrs: ast::AttrVec::new(),
        tokens: None,
    });
    ast::Stmt { id: ast::DUMMY_NODE_ID, kind: ast::StmtKind::Local(local), span: sp }
}
