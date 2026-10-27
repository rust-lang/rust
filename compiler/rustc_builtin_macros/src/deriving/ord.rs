use rustc_expand::base::ExtCtxt;
use rustc_span::{Span, sym};
use thin_vec::thin_vec;

use crate::deriving::generic::*;
use crate::deriving::partial_ord::{OrdlikeDerive, cmp_body, discr_data_order};
use crate::deriving::path_std;

pub(crate) fn expand_derive_ord(
    cx: &ExtCtxt<'_>,
    span: Span,
    item: &ast::Item,
    push: &mut dyn FnMut(Box<ast::Item>),
    is_const: bool,
) {
    let discr_then_data = discr_data_order(item);

    let trait_def = TraitDef {
        span,
        path: path_std!(cx, span, cmp::Ord),
        needs_copy_as_bound_if_packed: true,
        additional_bounds: SmallVec::new(),
        supports_unions: false,
        methods: smallvec![MethodDef {
            name: sym::cmp,
            generics: cx.empty_generics(span),
            explicit_self: true,
            nonself_args: smallvec![(sym::other, cx.ty_self_ref(span))],
            has_other_selflike_arg: true,
            ret_ty: cx.ty_path(path_std!(cx, span, cmp::Ordering)),
            attributes: thin_vec![cx.generated_attr(span, ast::GeneratedAttr::Inline)],
            fieldless_variants_strategy: FieldlessVariantsStrategy::Unify,
            body: Box::new(|cx, span, substr| {
                // Builds:
                //
                // match ::core::cmp::Ord::cmp(&self.x, &other.x) {
                //     ::std::cmp::Ordering::Equal =>
                //         ::core::cmp::Ord::cmp(&self.y, &other.y),
                //     cmp => cmp,
                // }
                let expr = cmp_body(cx, span, substr, discr_then_data, OrdlikeDerive::Ord);
                BlockOrExpr::new_expr(expr)
            }),
        }],
        is_const,
        ..
    };

    trait_def.expand(cx, item, push)
}
