use rustc_ast as ast;
use rustc_ast::{ItemKind, VariantData};
use rustc_errors::MultiSpan;
use rustc_expand::base::ExtCtxt;
use rustc_span::{Ident, Span, kw, sym};
use thin_vec::thin_vec;

use crate::deriving::generic::*;
use crate::diagnostics;
use crate::util::path;

/// Generate an implementation of the `From` trait, provided that `item`
/// is a struct or a tuple struct with exactly one field.
pub(crate) fn expand_deriving_from(
    cx: &ExtCtxt<'_>,
    span: Span,
    item: &ast::Item,
    push: &mut dyn FnMut(Box<ast::Item>),
    is_const: bool,
) {
    let err_span = || {
        let item_span = item.kind.ident().map(|ident| ident.span).unwrap_or(item.span);
        MultiSpan::from_spans(vec![span, item_span])
    };
    // `#[derive(From)]` is currently usable only on structs with exactly one field.
    let from_type = match &item.kind {
        ItemKind::Struct(_, _, data) => {
            if let [field] = data.fields() {
                field.ty.clone()
            } else {
                cx.dcx().emit_err(diagnostics::DeriveFromWrongFieldCount {
                    span: err_span(),
                    multiple_fields: data.fields().len() > 1,
                });
                return;
            }
        }
        ItemKind::Enum(_, _, _) | ItemKind::Union(_, _, _) => {
            cx.dcx().emit_err(diagnostics::DeriveFromWrongTarget {
                span: err_span(),
                kind: &format!("{} {}", item.kind.article(), item.kind.descr()),
            });
            return;
        }
        _ => cx.dcx().bug("Invalid derive(From) ADT input"),
    };

    let path =
        cx.std_path_all(span, path!(convert::From), vec![ast::GenericArg::Type(from_type.clone())]);

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
        additional_bounds: SmallVec::new(),
        supports_unions: false,
        methods: smallvec![MethodDef {
            name: sym::from,
            generics: cx.empty_generics(span),
            explicit_self: false,
            nonself_args: smallvec![(from_type, sym::value)],
            has_other_selflike_arg: false,
            ret_ty: cx.ty_self(span),
            attributes: thin_vec![cx.attr_word(sym::inline, span)],
            fieldless_variants_strategy: FieldlessVariantsStrategy::Default,
            combine_substructure: from_body,
        }],
        is_const,
        ..
    };

    from_trait_def.expand(cx, item, push);
}

fn from_body(cx: &ExtCtxt<'_>, span: Span, substr: Substructure<'_>) -> BlockOrExpr {
    let field = if let ItemKind::Struct(_, _, data) = &substr.item.kind
        && let [field] = data.fields()
    {
        field
    } else {
        unreachable!();
    };

    let self_kw = Ident::new(kw::SelfUpper, span);
    let expr = match substr.fields {
        StaticStruct(variant) => match variant {
            // Self { field: value }
            VariantData::Struct { .. } => cx.expr_struct_ident(
                span,
                self_kw,
                thin_vec![cx.field_imm(
                    span,
                    field.ident.unwrap(),
                    cx.expr_ident_sym(span, sym::value)
                )],
            ),
            // Self(value)
            VariantData::Tuple(_, _) => {
                cx.expr_call_ident(span, self_kw, thin_vec![cx.expr_ident_sym(span, sym::value)])
            }
            variant => {
                cx.dcx().bug(format!("Invalid derive(From) ADT variant: {variant:?}"));
            }
        },
        _ => cx.dcx().bug("Invalid derive(From) ADT input"),
    };
    BlockOrExpr::new_expr(expr)
}
