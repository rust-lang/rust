use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::path_std;

use rustc_ast::ptr::P;
use rustc_ast::{self as ast, Expr, MetaItem};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{sym, Ident};
use rustc_span::Span;

pub fn expand_deriving_eq(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
) {
    let span = cx.with_def_site_ctxt(span);
    let inline = cx.meta_word(span, sym::inline);
    let hidden = rustc_ast::attr::mk_nested_word_item(Ident::new(sym::hidden, span));
    let doc = rustc_ast::attr::mk_list_item(Ident::new(sym::doc, span), vec![hidden]);
    let no_coverage = cx.meta_word(span, sym::no_coverage);
    let attrs = vec![cx.attribute(inline), cx.attribute(doc), cx.attribute(no_coverage)];
    let trait_def = TraitDef {
        span,
        attributes: Vec::new(),
        path: path_std!(cmp::Eq),
        additional_bounds: Vec::new(),
        generics: Bounds::empty(),
        supports_unions: true,
        methods: vec![MethodDef {
            name: sym::assert_receiver_is_total_eq,
            generics: Bounds::empty(),
            explicit_self: borrowed_explicit_self(),
            args: vec![],
            ret_ty: nil_ty(),
            attributes: attrs,
            unify_fieldless_variants: true,
            combine_substructure: combine_substructure(Box::new(|a, b, c| {
                cs_total_eq_assert(a, b, c)
            })),
        }],
        associated_types: Vec::new(),
    };

    super::inject_impl_of_structural_trait(cx, span, item, path_std!(marker::StructuralEq), push);

    trait_def.expand_ext(cx, mitem, item, push, true)
}

fn cs_total_eq_assert(
    cx: &mut ExtCtxt<'_>,
    trait_span: Span,
    substr: &Substructure<'_>,
) -> P<Expr> {
    let mut stmts = Vec::new();
    let mut process_variant = |variant: &ast::VariantData| {
        for field in variant.fields() {
            // let _: AssertParamIsEq<FieldTy>;
            super::assert_ty_bounds(
                cx,
                &mut stmts,
                field.ty.clone(),
                field.span,
                &[sym::cmp, sym::AssertParamIsEq],
            );
        }
    };

    match *substr.fields {
        StaticStruct(vdata, ..) => {
            process_variant(vdata);
        }
        StaticEnum(enum_def, ..) => {
            for variant in &enum_def.variants {
                process_variant(&variant.data);
            }
        }
        _ => cx.span_bug(trait_span, "unexpected substructure in `derive(Eq)`"),
    }
    cx.expr_block(cx.block(trait_span, stmts))
}
