use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::path_std;

use rustc_ast::ptr::P;
use rustc_ast::{self as ast, Expr, LocalKind, MetaItem};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{sym, Ident};
use rustc_span::Span;

fn make_mut_borrow(cx: &mut ExtCtxt<'_>, sp: Span, expr: P<Expr>) -> P<Expr> {
    cx.expr(sp, ast::ExprKind::AddrOf(ast::BorrowKind::Ref, ast::Mutability::Mut, expr))
}

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
    let fmt = substr.nonself_args[0].clone();

    // Special fast path for unit variants. In the common case of an enum that is entirely unit
    // variants (i.e. a C-like enum), this fast path allows LLVM to eliminate the entire switch in
    // favor of a lookup table.
    if let ast::VariantData::Unit(..) = vdata {
        let fn_path_write_str = cx.std_path(&[sym::fmt, sym::Formatter, sym::write_str]);
        let expr = cx.expr_call_global(span, fn_path_write_str, vec![fmt, name]);
        let stmts = vec![cx.stmt_expr(expr)];
        let block = cx.block(span, stmts);
        return cx.expr_block(block);
    }

    if let ast::VariantData::Struct(..) = vdata {
        let mut name_exprs = Vec::with_capacity(fields.len() + 1);
        let mut value_exprs = Vec::with_capacity(fields.len());

        name_exprs.push(name);
        for field in fields {
            name_exprs.push(cx.expr_lit(
                field.span,
                ast::LitKind::Str(field.name.unwrap().name, ast::StrStyle::Cooked),
            ));

            // Use double indirection to make sure this works for unsized types
            let value_ref = cx.expr_addr_of(field.span, field.self_.clone());
            value_exprs.push(cx.expr_addr_of(field.span, value_ref));
        }

        // `let names: &'static _ = &["TheType", "field1", "field2"];`
        let lt_static = Some(cx.lifetime_static(span));
        let ty_static_ref = cx.ty_rptr(span, cx.ty_infer(span), lt_static, ast::Mutability::Not);
        let names_let = cx.stmt_let_ty(
            span,
            false,
            Ident::new(sym::names, span),
            Some(ty_static_ref),
            cx.expr_array_ref(span, name_exprs),
        );

        // `let values: &[&dyn Debug] = &[&&self.field1, &&self.field2];`
        let path_debug = cx.path_global(span, cx.std_path(&[sym::fmt, sym::Debug]));
        let ty_dyn_debug = cx.ty(
            span,
            ast::TyKind::TraitObject(vec![cx.trait_bound(path_debug)], ast::TraitObjectSyntax::Dyn),
        );
        let ty_slice = cx.ty(
            span,
            ast::TyKind::Slice(cx.ty_rptr(span, ty_dyn_debug, None, ast::Mutability::Not)),
        );
        let values_let = cx.stmt_let_ty(
            span,
            false,
            Ident::new(sym::values, span),
            Some(cx.ty_rptr(span, ty_slice, None, ast::Mutability::Not)),
            cx.expr_array_ref(span, value_exprs),
        );

        // `fmt::DebugStruct::internal_debug_from_slices(fmt, names, values)`
        let fn_path_debug_internal =
            cx.std_path(&[sym::fmt, sym::DebugStruct, sym::internal_debug_from_slices]);
        let args = vec![
            fmt,
            cx.expr_ident(span, Ident::new(sym::names, span)),
            cx.expr_ident(span, Ident::new(sym::values, span)),
        ];
        let expr = cx.expr_call_global(span, fn_path_debug_internal, args);

        let block = cx.block(span, vec![names_let, values_let, cx.stmt_expr(expr)]);
        return cx.expr_block(block);
    }

    let builder = Ident::new(sym::debug_trait_builder, span);
    let builder_expr = cx.expr_ident(span, builder);

    let mut stmts = Vec::with_capacity(fields.len() + 2);
    let fn_path_finish;
    match vdata {
        ast::VariantData::Unit(..) => {
            cx.span_bug(span, "unit variants should have been handled above");
        }
        ast::VariantData::Tuple(..) => {
            // tuple struct/"normal" variant
            let fn_path_debug_tuple = cx.std_path(&[sym::fmt, sym::Formatter, sym::debug_tuple]);
            let expr = cx.expr_call_global(span, fn_path_debug_tuple, vec![fmt, name]);
            let expr = make_mut_borrow(cx, span, expr);
            stmts.push(cx.stmt_let(span, false, builder, expr));

            for field in fields {
                // Use double indirection to make sure this works for unsized types
                let field = cx.expr_addr_of(field.span, field.self_.clone());
                let field = cx.expr_addr_of(field.span, field);

                let fn_path_field = cx.std_path(&[sym::fmt, sym::DebugTuple, sym::field]);
                let expr =
                    cx.expr_call_global(span, fn_path_field, vec![builder_expr.clone(), field]);

                // Use `let _ = expr;` to avoid triggering the
                // unused_results lint.
                stmts.push(stmt_let_underscore(cx, span, expr));
            }

            fn_path_finish = cx.std_path(&[sym::fmt, sym::DebugTuple, sym::finish]);
        }
        ast::VariantData::Struct(..) => {
            cx.span_bug(span, "structs should have been handled above");

            /*
            // normal struct/struct variant
            let fn_path_debug_struct = cx.std_path(&[sym::fmt, sym::Formatter, sym::debug_struct]);
            let expr = cx.expr_call_global(span, fn_path_debug_struct, vec![fmt, name]);
            let expr = make_mut_borrow(cx, span, expr);
            stmts.push(cx.stmt_let(DUMMY_SP, false, builder, expr));

            for field in fields {
                let name = cx.expr_lit(
                    field.span,
                    ast::LitKind::Str(field.name.unwrap().name, ast::StrStyle::Cooked),
                );

                // Use double indirection to make sure this works for unsized types
                let fn_path_field = cx.std_path(&[sym::fmt, sym::DebugStruct, sym::field]);
                let field = cx.expr_addr_of(field.span, field.self_.clone());
                let field = cx.expr_addr_of(field.span, field);
                let expr = cx.expr_call_global(
                    span,
                    fn_path_field,
                    vec![builder_expr.clone(), name, field],
                );
                stmts.push(stmt_let_underscore(cx, span, expr));
            }
            fn_path_finish = cx.std_path(&[sym::fmt, sym::DebugStruct, sym::finish]);
            */
        }
    }

    let expr = cx.expr_call_global(span, fn_path_finish, vec![builder_expr]);

    stmts.push(cx.stmt_expr(expr));
    let block = cx.block(span, stmts);
    cx.expr_block(block)
}

fn stmt_let_underscore(cx: &mut ExtCtxt<'_>, sp: Span, expr: P<ast::Expr>) -> ast::Stmt {
    let local = P(ast::Local {
        pat: cx.pat_wild(sp),
        ty: None,
        id: ast::DUMMY_NODE_ID,
        kind: LocalKind::Init(expr),
        span: sp,
        attrs: ast::AttrVec::new(),
        tokens: None,
    });
    ast::Stmt { id: ast::DUMMY_NODE_ID, kind: ast::StmtKind::Local(local), span: sp }
}
