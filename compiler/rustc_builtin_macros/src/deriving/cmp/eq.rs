use rustc_ast::{self as ast, MetaItem, Safety};
use rustc_data_structures::fx::FxHashSet;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::{Ident, Span, kw, sym};
use thin_vec::{ThinVec, thin_vec};

use crate::deriving::generic::*;
use crate::deriving::path_std;

pub(crate) fn expand_deriving_eq(
    cx: &ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    is_const: bool,
) {
    let span = cx.with_def_site_ctxt(span);
    let mut impl_generics = ast::Generics::default();

    let trait_def = TraitDef {
        span,
        path: path_std!(cmp::Eq),
        skip_path_as_bound: false,
        needs_copy_as_bound_if_packed: true,
        additional_bounds: Vec::new(),
        supports_unions: true,
        methods: Vec::new(),
        associated_types: Vec::new(),
        is_const,
        is_staged_api_crate: cx.ecfg.features.staged_api(),
        safety: Safety::Default,
        document: true,
    };
    trait_def.expand_ext(
        cx,
        mitem,
        item,
        &mut |a| {
            let Annotatable::Item(item) = &a else {
                unreachable!("should have emitted an Item in trait_def.expand_ext");
            };
            let ast::ItemKind::Impl(imp) = &item.kind else {
                unreachable!("should have emitted an Impl in trait_def.expand_ext");
            };
            impl_generics = imp.generics.clone();
            push(a)
        },
        true,
    );

    let const_body = cx.block(
        span,
        thin_vec![cx.stmt_item(span, assert_fields_are_eq_fn(cx, span, item, impl_generics),)],
    );

    let unit_ty = cx.ty(span, ast::TyKind::Tup(ThinVec::new()));
    let body = cx.expr_block(const_body);
    let konst = cx.item_const(
        span,
        Ident::new(kw::Underscore, span),
        unit_ty,
        ast::ConstItemRhsKind::new_body(body),
    );
    push(Annotatable::Item(konst));
}

fn assert_fields_are_eq_fn(
    cx: &ExtCtxt<'_>,
    span: Span,
    item: &Annotatable,
    fn_generics: ast::Generics,
) -> Box<ast::Item> {
    let stmts = eq_assert_stmts_from_item(cx, span, item);

    cx.item(
        span,
        thin_vec![
            cx.attr_nested_word(sym::doc, sym::hidden, span),
            cx.attr_nested_word(sym::coverage, sym::off, span),
        ],
        ast::ItemKind::Fn(Box::new(ast::Fn {
            defaultness: ast::Defaultness::Implicit,
            ident: Ident::new(sym::assert_fields_are_eq, span),
            generics: fn_generics,
            sig: ast::FnSig {
                header: ast::FnHeader {
                    constness: ast::Const::Yes(span),
                    coroutine_kind: None,
                    safety: ast::Safety::Default,
                    ext: ast::Extern::None,
                },
                decl: cx.fn_decl(ThinVec::new(), ast::FnRetTy::Default(span)),
                span,
            },
            contract: None,
            define_opaque: None,
            body: Some(cx.block(span, stmts)),
            eii_impls: ThinVec::new(),
        })),
    )
}

fn eq_assert_stmts_from_item(
    cx: &ExtCtxt<'_>,
    span: Span,
    item: &Annotatable,
) -> ThinVec<ast::Stmt> {
    let mut stmts = ThinVec::new();
    let mut seen_type_names = FxHashSet::default();
    let mut process_variant = |variant: &ast::VariantData| {
        for field in variant.fields() {
            // This basic redundancy checking only prevents duplication of
            // assertions like `AssertParamIsEq<Foo>` where the type is a
            // simple name. That's enough to get a lot of cases, though.
            if let Some(name) = field.ty.kind.is_simple_path()
                && !seen_type_names.insert(name)
            {
                // Already produced an assertion for this type.
            } else {
                // let _: AssertParamIsEq<FieldTy>;
                super::assert_ty_bounds(
                    cx,
                    &mut stmts,
                    field.ty.clone(),
                    field.span,
                    &[sym::cmp, sym::AssertParamIsEq],
                );
            }
        }
    };
    match item {
        Annotatable::Item(item) => match &item.kind {
            ast::ItemKind::Struct(_, _, vdata) => {
                process_variant(vdata);
            }
            ast::ItemKind::Enum(_, _, enum_def) => {
                for variant in &enum_def.variants {
                    process_variant(&variant.data);
                }
            }
            ast::ItemKind::Union(_, _, vdata) => {
                process_variant(vdata);
            }
            _ => cx.dcx().span_bug(span, "unexpected item in `derive(Eq)`"),
        },
        _ => cx.dcx().span_bug(span, "unexpected item in `derive(Eq)`"),
    }
    stmts
}
