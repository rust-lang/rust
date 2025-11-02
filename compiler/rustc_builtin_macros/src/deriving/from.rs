use rustc_ast as ast;
use rustc_ast::{ItemKind, VariantData};
use rustc_errors::MultiSpan;
use rustc_expand::base::{Annotatable, DummyResult, ExtCtxt};
use rustc_span::{Ident, Span, kw, sym};
use thin_vec::thin_vec;

use crate::deriving::generic::ty::{Bounds, Path, PathKind, Ty};
use crate::deriving::generic::{
    BlockOrExpr, FieldlessVariantsStrategy, MethodDef, SubstructureFields, TraitDef,
    combine_substructure,
};
use crate::deriving::pathvec_std;
use crate::errors;

/// Generate an implementation of the `From` trait, provided that `item`
/// is a struct or a tuple struct with exactly one field.
pub(crate) fn expand_deriving_from(
    cx: &ExtCtxt<'_>,
    span: Span,
    mitem: &ast::MetaItem,
    annotatable: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    is_const: bool,
) {
    let Annotatable::Item(item) = &annotatable else {
        cx.dcx().bug("derive(From) used on something else than an item");
    };

    let err_span = || {
        let item_span = item.kind.ident().map(|ident| ident.span).unwrap_or(item.span);
        MultiSpan::from_spans(vec![span, item_span])
    };

    // `#[derive(From)]` is currently usable only on structs with exactly one field.
    let field = match &item.kind {
        ItemKind::Struct(_, _, data) => {
            if let [field] = data.fields() {
                Ok(field.clone())
            } else {
                let guar = cx.dcx().emit_err(errors::DeriveFromWrongFieldCount {
                    span: err_span(),
                    multiple_fields: data.fields().len() > 1,
                });
                Err(guar)
            }
        }
        ItemKind::Enum(_, _, _) | ItemKind::Union(_, _, _) => {
            let guar = cx.dcx().emit_err(errors::DeriveFromWrongTarget {
                span: err_span(),
                kind: &format!("{} {}", item.kind.article(), item.kind.descr()),
            });
            Err(guar)
        }
        _ => cx.dcx().bug("Invalid derive(From) ADT input"),
    };

    let from_type = Ty::AstTy(match field {
        Ok(ref field) => field.ty.clone(),
        Err(guar) => cx.ty(span, ast::TyKind::Err(guar)),
    });

    let path =
        Path::new_(pathvec_std!(convert::From), vec![Box::new(from_type.clone())], PathKind::Std);

    // Generate code like this:
    //
    // struct S(u32);
    // #[automatically_derived]
    // impl ::core::convert::From<u32> for S {
    //     #[inline]
    //     fn from(value: u32) -> S {
    //         Self(value)
    //     }
    // }
    let from_trait_def = TraitDef {
        span,
        path,
        skip_path_as_bound: true,
        needs_copy_as_bound_if_packed: false,
        additional_bounds: Vec::new(),
        supports_unions: false,
        methods: vec![MethodDef {
            name: sym::from,
            generics: Bounds { bounds: vec![] },
            explicit_self: false,
            nonself_args: vec![(from_type, sym::value)],
            ret_ty: Ty::Self_,
            attributes: thin_vec![cx.attr_word(sym::inline, span)],
            fieldless_variants_strategy: FieldlessVariantsStrategy::Default,
            combine_substructure: combine_substructure(Box::new(|cx, span, substructure| {
                let field = match field {
                    Ok(ref field) => field,
                    Err(guar) => {
                        return BlockOrExpr::new_expr(DummyResult::raw_expr(span, Some(guar)));
                    }
                };

                let self_kw = Ident::new(kw::SelfUpper, span);
                let expr: Box<ast::Expr> = match substructure.fields {
                    SubstructureFields::StaticStruct(variant, _) => match variant {
                        // Self { field: value }
                        VariantData::Struct { .. } => cx.expr_struct_ident(
                            span,
                            self_kw,
                            thin_vec![cx.field_imm(
                                span,
                                field.ident.unwrap(),
                                cx.expr_ident(span, Ident::new(sym::value, span))
                            )],
                        ),
                        // Self(value)
                        VariantData::Tuple(_, _) => cx.expr_call_ident(
                            span,
                            self_kw,
                            thin_vec![cx.expr_ident(span, Ident::new(sym::value, span))],
                        ),
                        variant => {
                            cx.dcx().bug(format!("Invalid derive(From) ADT variant: {variant:?}"));
                        }
                    },
                    _ => cx.dcx().bug("Invalid derive(From) ADT input"),
                };
                BlockOrExpr::new_expr(expr)
            })),
        }],
        associated_types: Vec::new(),
        is_const,
        is_staged_api_crate: cx.ecfg.features.staged_api(),
    };

    from_trait_def.expand(cx, mitem, annotatable, push);
}
