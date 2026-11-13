use rustc_ast::{self as ast};
use rustc_data_structures::fx::FxHashSet;
use rustc_expand::base::ExtCtxt;
use rustc_span::{Span, sym};
use thin_vec::{ThinVec, thin_vec};

use crate::deriving::generic::*;
use crate::deriving::path_std;

pub(crate) fn expand_derive_eq(
    cx: &ExtCtxt<'_>,
    span: Span,
    item: &ast::Item,
    push: &mut dyn FnMut(Box<ast::Item>),
    is_const: bool,
) {
    let span = cx.with_def_site_ctxt(span);

    let trait_def = TraitDef {
        span,
        path: path_std!(cx, span, cmp::Eq),
        needs_copy_as_bound_if_packed: true,
        additional_bounds: SmallVec::new(),
        supports_unions: true,
        methods: smallvec![MethodDef {
            name: sym::assert_fields_are_eq,
            generics: cx.empty_generics(span),
            args: thin_vec![cx.param_self_ref(span)],
            ret_ty: cx.ty_unit(span),
            attributes: thin_vec![
                // This method will never be called, so doing codegen etc. for it is unnecessary.
                // We prevent this by adding `#[inline]`, which improves compile-time.
                cx.generated_attr(span, ast::GeneratedAttr::Inline),
                cx.generated_attr(span, ast::GeneratedAttr::DocHidden),
                cx.generated_attr(span, ast::GeneratedAttr::CoverageOff),
            ],
            body: MethodBody::Directly(assert_fields_eq_body(cx, item)),
        }],
        is_const,
        ..
    };
    trait_def.expand(cx, item, push);
}

fn assert_fields_eq_body(cx: &ExtCtxt<'_>, item: &ast::Item) -> BlockOrExpr {
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

    match &item.kind {
        ast::ItemKind::Struct(.., vdata) | ast::ItemKind::Union(.., vdata) => {
            process_variant(vdata);
        }
        ast::ItemKind::Enum(.., enum_def) => {
            for variant in &enum_def.variants {
                process_variant(&variant.data);
            }
        }
        _ => unreachable!(),
    }
    BlockOrExpr::new_stmts(stmts)
}
