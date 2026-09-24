use rustc_ast::{Expr, ItemKind, Safety, ast};
use rustc_expand::base::ExtCtxt;
use rustc_span::{Ident, Span, kw, sym};
use thin_vec::thin_vec;

use crate::deriving::generic::*;
use crate::deriving::{call_discriminant_value, new_path, path_std, pathvec};

pub(crate) fn expand_deriving_partial_ord(
    cx: &ExtCtxt<'_>,
    span: Span,
    item: &ast::Item,
    push: &mut dyn FnMut(Box<ast::Item>),
    is_const: bool,
) {
    let ordering_ty = cx.ty_path(path_std!(cx, span, cmp::Ordering));
    let ret_ty = cx.ty_path(new_path(cx, span, pathvec!(option::Option), vec![ordering_ty]));

    // Order in which to perform matching
    let discr_then_data = discr_data_order(item);

    let container_id = cx.current_expansion.id.expn_data().parent.expect_local();
    let has_derive_ord = cx.resolver.has_derive_ord(container_id);
    let default_substructure =
        combine_substructure(|cx, span, substr| cs_partial_cmp(cx, span, substr, discr_then_data));
    let simple_substructure = combine_substructure(|cx, span, _| {
        cs_partial_cmp_simple(cx, span, cx.expr_ident_sym(span, sym::other))
    });
    let is_simple = match &item.kind {
        // For unit structs/zero-variant enums, the default generated code is better.
        ItemKind::Struct(.., ast::VariantData::Unit(..)) => false,
        // Also for single fieldless variant enum
        ItemKind::Enum(.., enum_def) if enum_def.variants.is_empty() => false,
        ItemKind::Enum(.., enum_def)
            if enum_def.variants.len() == 1
                && matches!(enum_def.variants[0].data, ast::VariantData::Unit(..)) =>
        {
            false
        }
        ItemKind::Struct(_, ast::Generics { params, .. }, _)
        | ItemKind::Enum(_, ast::Generics { params, .. }, _)
            if has_derive_ord
                && !params
                    .iter()
                    .any(|param| matches!(param.kind, ast::GenericParamKind::Type { .. })) =>
        {
            true
        }
        _ => false,
    };

    let partial_cmp_def = MethodDef {
        name: sym::partial_cmp,
        generics: cx.empty_generics(span),
        explicit_self: true,
        nonself_args: smallvec![(cx.ty_self_ref(span), sym::other)],
        has_other_selflike_arg: true,
        ret_ty,
        attributes: thin_vec![cx.attr_word(sym::inline, span)],
        fieldless_variants_strategy: FieldlessVariantsStrategy::Unify,
        combine_substructure: if is_simple { simple_substructure } else { default_substructure },
    };

    let trait_def = TraitDef {
        span,
        path: path_std!(cx, span, cmp::PartialOrd),
        skip_path_as_bound: false,
        needs_copy_as_bound_if_packed: true,
        additional_bounds: smallvec![],
        supports_unions: false,
        methods: smallvec![partial_cmp_def],
        is_const,
        safety: Safety::Default,
        document: true,
    };
    trait_def.expand_ext(cx, item, push, is_simple)
}

pub(crate) fn discr_data_order(item: &ast::Item) -> bool {
    if let ItemKind::Enum(_, _, def) = &item.kind {
        let dataful: Vec<bool> = def.variants.iter().map(|v| !v.data.fields().is_empty()).collect();
        match dataful.iter().filter(|&&b| b).count() {
            // No data, placing the discriminant check first makes codegen simpler
            0 => true,
            1..=2 => false,
            _ => (0..dataful.len() - 1).any(|i| {
                if dataful[i]
                    && let Some(idx) = dataful[i + 1..].iter().position(|v| *v)
                {
                    idx >= 2
                } else {
                    false
                }
            }),
        }
    } else {
        true
    }
}

// Special case for the type deriving both `PartialOrd` and `Ord`. Builds:
// ```
// Some(::core::cmp::Ord::cmp(self, other))
// ```
fn cs_partial_cmp_simple(cx: &ExtCtxt<'_>, span: Span, other_expr: Box<ast::Expr>) -> BlockOrExpr {
    let ord_cmp_path = cx.std_path(&[sym::cmp, sym::Ord, sym::cmp]);
    let cmp_expr =
        cx.expr_call_global(span, ord_cmp_path, thin_vec![cx.expr_self(span), other_expr]);
    BlockOrExpr::new_expr(cx.expr_some(span, cmp_expr))
}

fn cs_partial_cmp(
    cx: &ExtCtxt<'_>,
    span: Span,
    substr: Substructure<'_>,
    discr_then_data: bool,
) -> BlockOrExpr {
    // Builds:
    //
    // match ::core::cmp::PartialOrd::partial_cmp(&self.x, &other.x) {
    //     ::core::option::Option::Some(::core::cmp::Ordering::Equal) =>
    //         ::core::cmp::PartialOrd::partial_cmp(&self.y, &other.y),
    //     cmp => cmp,
    // }
    let expr = cmp_body(cx, span, substr, discr_then_data, OrdlikeDerive::PartialOrd);
    BlockOrExpr::new_expr(expr)
}

#[derive(PartialEq)]
pub(crate) enum OrdlikeDerive {
    PartialOrd,
    Ord,
}

pub(crate) fn cmp_body(
    cx: &ExtCtxt<'_>,
    span: Span,
    substructure: Substructure<'_>,
    discr_then_data: bool,
    derive: OrdlikeDerive,
) -> Box<Expr> {
    let is_partial_ord = derive == OrdlikeDerive::PartialOrd;
    let method_path = if is_partial_ord {
        cx.std_path(&[sym::cmp, sym::PartialOrd, sym::partial_cmp])
    } else {
        cx.std_path(&[sym::cmp, sym::Ord, sym::cmp])
    };
    let equal_path = cx.path_global(span, cx.std_path(&[sym::cmp, sym::Ordering, sym::Equal]));

    // The combination of two field expressions. E.g. for `Ord::cmp` this
    // is something like `<field1 comparison> && <field2 comparison>`.
    let combine = |span, mut expr1: Box<Expr>, expr2| {
        // For `PartialOrd` (`Ord` works the same but without the `Some` wrapping),
        // when the item is an enum, this expands to
        // ```
        // match (expr2) {
        //     Some(Ordering::Equal) => expr1,
        //     cmp => cmp
        // }
        // ```
        // where `expr2` is `partial_cmp(self_discr, other_discr)`, and `expr1` is a `match`
        // against the enum variants. This means that we begin by comparing the enum discriminants,
        // before either inspecting their contents (if they match), or returning
        // the `cmp::Ordering` of comparing the enum discriminants.
        // ```
        // match partial_cmp(self_discr, other_discr) {
        //     Some(Ordering::Equal) => match (self, other)  {
        //         (Self::A(self_0), Self::A(other_0)) => partial_cmp(self_0, other_0),
        //         (Self::B(self_0), Self::B(other_0)) => partial_cmp(self_0, other_0),
        //         _ => Some(Ordering::Equal)
        //     }
        //     cmp => cmp
        // }
        // ```
        // If we have any certain enum layouts, flipping this results in better codegen
        // ```
        // match (self, other) {
        //     (Self::A(self_0), Self::A(other_0)) => partial_cmp(self_0, other_0),
        //     _ => partial_cmp(self_discr, other_discr)
        // }
        // ```
        // Reference: https://github.com/rust-lang/rust/pull/103659#issuecomment-1328126354

        if !discr_then_data
            && let ast::ExprKind::Match(_, arms, _) = &mut expr1.kind
            && let Some(last) = arms.last_mut()
            && let ast::PatKind::Wild = last.pat.kind
        {
            last.body = Some(expr2);
            expr1
        } else {
            let eq_pat = cx.pat_path(span, equal_path.clone());
            let eq_arm = cx.arm(
                span,
                if is_partial_ord { cx.pat_some(span, eq_pat) } else { eq_pat },
                expr1,
            );
            let cmp_ident = Ident::new(sym::cmp, span);
            let neq_arm =
                cx.arm(span, cx.pat_ident(span, cmp_ident), cx.expr_ident(span, cmp_ident));
            cx.expr_match(span, expr2, thin_vec![eq_arm, neq_arm])
        }
    };

    match substructure {
        EnumMatching(.., all_fields) | Struct(_, all_fields) => {
            let op = |old, field: FieldInfo| {
                // The basic case: a field expression for one or more selflike args. E.g.
                // for `Ord::cmp` this is something like `Ord::cmp(&self.x, &other.x)`.
                let other_expr =
                    field.other_selflike_expr.expect("not exactly 2 arguments in `derive`");
                let args = thin_vec![field.self_expr, other_expr];
                let new = cx.expr_call_global(field.span, method_path.clone(), args);
                match old {
                    Some(old) => Some(combine(field.span, old, new)),
                    None => Some(new),
                }
            };

            all_fields.into_iter().rfold(None, op).unwrap_or_else(|| {
                // The fallback case for a struct or enum variant with no fields.
                let mut expr = cx.expr_path(equal_path);
                if is_partial_ord {
                    expr = cx.expr_some(span, expr)
                };
                expr
            })
        }
        EnumDiscr(match_expr) => {
            let self_expr = cx.expr_addr_of(span, call_discriminant_value(cx, span, kw::SelfLower));
            let other_expr = cx.expr_addr_of(span, call_discriminant_value(cx, span, sym::other));
            let args = thin_vec![self_expr, other_expr];
            let discr_check_expr = cx.expr_call_global(span, method_path, args);
            if let Some(match_expr) = match_expr {
                combine(span, match_expr, discr_check_expr)
            } else {
                discr_check_expr
            }
        }
        _ => cx.dcx().span_bug(span, "unexpected substructure in `derive`"),
    }
}
