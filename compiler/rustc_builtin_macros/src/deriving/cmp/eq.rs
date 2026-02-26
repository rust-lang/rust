use rustc_ast::{self as ast, MetaItem, Safety};
use rustc_data_structures::fx::FxHashSet;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::{Ident, Span, sym};
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
    let mut impl_self_ty = None;

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
        &mut |mut a| {
            let Annotatable::Item(item) = &mut a else {
                unreachable!("should have emitted an Item in trait_def.expand_ext");
            };
            let ast::ItemKind::Impl(imp) = &mut item.kind else {
                unreachable!("should have emitted an Impl in trait_def.expand_ext");
            };
            impl_generics = imp.generics.clone();
            impl_self_ty = Some(imp.self_ty.clone());
            respan_generics_for_derive(&mut imp.generics, span);
            push(a)
        },
        true,
    );

    let assert_stmts = eq_assert_stmts_from_item(cx, span, item);

    // Skip generating `assert_fields_are_eq` impl if there are no assertions to make
    if assert_stmts.is_empty() {
        return;
    }

    let impl_self_ty = impl_self_ty
        .unwrap_or_else(|| cx.dcx().span_bug(span, "missing self type in `derive(Eq)`"));
    push(Annotatable::Item(assert_fields_are_eq_impl(
        cx,
        span,
        strip_const_trait_bounds_from_generics(impl_generics),
        impl_self_ty,
        assert_stmts,
    )));
}

fn respan_generics_for_derive(generics: &mut ast::Generics, span: Span) {
    for predicate in &mut generics.where_clause.predicates {
        predicate.span = span.with_ctxt(predicate.span.ctxt());

        match &mut predicate.kind {
            ast::WherePredicateKind::BoundPredicate(bound_predicate) => {
                bound_predicate.bounded_ty.span =
                    span.with_ctxt(bound_predicate.bounded_ty.span.ctxt());

                for bound in &mut bound_predicate.bounds {
                    match bound {
                        ast::GenericBound::Trait(poly_trait_ref) => {
                            poly_trait_ref.span = span.with_ctxt(poly_trait_ref.span.ctxt());
                        }
                        _ => {}
                    }
                }
            }
            ast::WherePredicateKind::RegionPredicate(region_predicate) => {
                for bound in &mut region_predicate.bounds {
                    match bound {
                        ast::GenericBound::Trait(poly_trait_ref) => {
                            poly_trait_ref.span = span.with_ctxt(poly_trait_ref.span.ctxt());
                        }
                        _ => {}
                    }
                }
            }
            ast::WherePredicateKind::EqPredicate(eq_predicate) => {
                eq_predicate.lhs_ty.span = span.with_ctxt(eq_predicate.lhs_ty.span.ctxt());
                eq_predicate.rhs_ty.span = span.with_ctxt(eq_predicate.rhs_ty.span.ctxt());
            }
        }
    }
    generics.where_clause.span = span.with_ctxt(generics.where_clause.span.ctxt());
    generics.span = span.with_ctxt(generics.span.ctxt());
}

fn assert_fields_are_eq_impl(
    cx: &ExtCtxt<'_>,
    span: Span,
    impl_generics: ast::Generics,
    impl_self_ty: Box<ast::Ty>,
    assert_stmts: ThinVec<ast::Stmt>,
) -> Box<ast::Item> {
    cx.item(
        span,
        ast::AttrVec::new(),
        ast::ItemKind::Impl(ast::Impl {
            generics: impl_generics,
            of_trait: None,
            constness: ast::Const::No,
            self_ty: impl_self_ty,
            items: thin_vec![Box::new(ast::AssocItem {
                id: ast::DUMMY_NODE_ID,
                attrs: thin_vec![
                    cx.attr_nested_word(sym::doc, sym::hidden, span),
                    cx.attr_nested_word(sym::coverage, sym::off, span),
                ],
                span,
                vis: ast::Visibility {
                    span: span.shrink_to_lo(),
                    kind: ast::VisibilityKind::Inherited,
                    tokens: None,
                },
                kind: ast::AssocItemKind::Fn(Box::new(ast::Fn {
                    defaultness: ast::Defaultness::Implicit,
                    ident: Ident::new(sym::assert_fields_are_eq, span),
                    generics: ast::Generics { span, ..Default::default() },
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
                    body: Some(cx.block(span, assert_stmts)),
                    eii_impls: ThinVec::new(),
                })),
                tokens: None,
            })],
        }),
    )
}

fn strip_const_trait_bounds_from_generics(mut generics: ast::Generics) -> ast::Generics {
    for param in &mut generics.params {
        if let ast::GenericParamKind::Type { .. } = &param.kind {
            strip_constness_from_bounds(&mut param.bounds);
        }
    }

    for predicate in &mut generics.where_clause.predicates {
        if let ast::WherePredicateKind::BoundPredicate(bound_predicate) = &mut predicate.kind {
            strip_constness_from_bounds(&mut bound_predicate.bounds);
        }
    }

    generics
}

fn strip_constness_from_bounds(bounds: &mut [ast::GenericBound]) {
    for bound in bounds {
        if let ast::GenericBound::Trait(poly_trait_ref) = bound {
            poly_trait_ref.modifiers.constness = ast::BoundConstness::Never;
        }
    }
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
