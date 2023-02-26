use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::path_std;

use rustc_ast::{self as ast, MetaItem};
use rustc_data_structures::fx::FxHashSet;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::sym;
use rustc_span::Span;
use thin_vec::{thin_vec, ThinVec};

pub fn expand_deriving_eq(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    is_const: bool,
) {
    let span = cx.with_def_site_ctxt(span);
    let attrs = thin_vec![
        cx.attr_word(sym::inline, span),
        cx.attr_nested_word(sym::doc, sym::hidden, span),
        cx.attr_word(sym::no_coverage, span)
    ];
    let trait_def = TraitDef {
        span,
        path: path_std!(cmp::Eq),
        skip_path_as_bound: false,
        needs_copy_as_bound_if_packed: true,
        additional_bounds: Vec::new(),
        supports_unions: true,
        methods: vec![MethodDef {
            name: sym::assert_receiver_is_total_eq,
            generics: Bounds::empty(),
            explicit_self: true,
            nonself_args: vec![],
            ret_ty: Unit,
            attributes: attrs,
            fieldless_variants_strategy: FieldlessVariantsStrategy::Unify,
            combine_substructure: combine_substructure(Box::new(|a, b, c| {
                cs_total_eq_assert(a, b, c)
            })),
        }],
        associated_types: Vec::new(),
        is_const,
    };

    super::inject_impl_of_structural_trait(cx, span, item, path_std!(marker::StructuralEq), push);

    trait_def.expand_ext(cx, mitem, item, push, true)
}

fn cs_total_eq_assert(
    cx: &mut ExtCtxt<'_>,
    trait_span: Span,
    substr: &Substructure<'_>,
) -> BlockOrExpr {
    let mut stmts = ThinVec::new();
    let mut seen_type_names = FxHashSet::default();
    let mut process_variant = |variant: &ast::VariantData| {
        for field in variant.fields() {
            // This basic redundancy checking only prevents duplication of
            // assertions like `AssertParamIsEq<Foo>` where the type is a
            // simple name. That's enough to get a lot of cases, though.
            if let Some(name) = field.ty.kind.is_simple_path() && !seen_type_names.insert(name) {
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
    BlockOrExpr::new_stmts(stmts)
}
