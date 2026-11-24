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
    // We can skip the assertions for all fields that are named in the generics,
    // as we already have a `T: Eq` bound for each of them.
    // FIXME: this could be extended to also skip `T::AssocType` as that also gets an impl bound.
    let mut seen_type_names =
        FxHashSet::from_iter(item.opt_generics().unwrap().params.iter().map(|p| p.ident.name));
    for field in cx.fields_iter(item) {
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

    BlockOrExpr::new_stmts(stmts)
}
