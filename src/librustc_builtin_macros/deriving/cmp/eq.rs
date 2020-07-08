use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::path_std;

use rustc_ast::ast::{self, Expr, GenericArg, MetaItem};
use rustc_ast::ptr::P;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::Span;

pub fn expand_deriving_eq(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
) {
    let inline = cx.meta_word(span, sym::inline);
    let hidden = rustc_ast::attr::mk_nested_word_item(Ident::new(sym::hidden, span));
    let doc = rustc_ast::attr::mk_list_item(Ident::new(sym::doc, span), vec![hidden]);
    let attrs = vec![cx.attribute(inline), cx.attribute(doc)];
    let trait_def = TraitDef {
        span,
        attributes: Vec::new(),
        path: path_std!(cx, cmp::Eq),
        additional_bounds: Vec::new(),
        generics: LifetimeBounds::empty(),
        is_unsafe: false,
        supports_unions: true,
        methods: vec![MethodDef {
            name: sym::assert_receiver_is_total_eq,
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

    super::inject_impl_of_structural_trait(
        cx,
        span,
        item,
        path_std!(cx, marker::StructuralEq),
        push,
    );

    trait_def.expand_ext(cx, mitem, item, push, true)
}

fn cs_total_eq_assert(
    cx: &mut ExtCtxt<'_>,
    trait_span: Span,
    substr: &Substructure<'_>,
) -> P<Expr> {
    fn assert_ty_bounds(
        cx: &mut ExtCtxt<'_>,
        stmts: &mut Vec<ast::Stmt>,
        ty: P<ast::Ty>,
        span: Span,
        helper_name: &str,
    ) {
        // Generate statement `let _: helper_name<ty>;`,
        // set the expn ID so we can use the unstable struct.
        let span = cx.with_def_site_ctxt(span);
        let assert_path = cx.path_all(
            span,
            true,
            cx.std_path(&[sym::cmp, Symbol::intern(helper_name)]),
            vec![GenericArg::Type(ty)],
        );
        stmts.push(cx.stmt_let_type_only(span, cx.ty_path(assert_path)));
    }
    fn process_variant(
        cx: &mut ExtCtxt<'_>,
        stmts: &mut Vec<ast::Stmt>,
        variant: &ast::VariantData,
    ) {
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
                process_variant(cx, &mut stmts, &variant.data);
            }
        }
        _ => cx.span_bug(trait_span, "unexpected substructure in `derive(Eq)`"),
    }
    cx.expr_block(cx.block(trait_span, stmts))
}
