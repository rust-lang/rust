use rustc_ast::Safety;
use rustc_expand::base::ExtCtxt;
use rustc_span::{Span, sym};
use thin_vec::thin_vec;

use crate::deriving::generic::*;
use crate::deriving::partial_ord::{OrdlikeDerive, cmp_body, discr_data_order};
use crate::deriving::path_std;

pub(crate) fn expand_deriving_ord(
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
        skip_path_as_bound: false,
        needs_copy_as_bound_if_packed: true,
        additional_bounds: SmallVec::new(),
        supports_unions: false,
        methods: smallvec![MethodDef {
            name: sym::cmp,
            generics: cx.empty_generics(span),
            explicit_self: true,
            nonself_args: smallvec![(cx.ty_self_ref(span), sym::other)],
            has_other_selflike_arg: true,
            ret_ty: cx.ty_path(path_std!(cx, span, cmp::Ordering)),
            attributes: thin_vec![cx.attr_word(sym::inline, span)],
            fieldless_variants_strategy: FieldlessVariantsStrategy::Unify,
            combine_substructure: combine_substructure(|cx, span, substr| cs_cmp(
                cx,
                span,
                substr,
                discr_then_data
            )),
        }],
        is_const,
        safety: Safety::Default,
        document: true,
    };

    trait_def.expand(cx, item, push)
}

pub(crate) fn cs_cmp(
    cx: &ExtCtxt<'_>,
    span: Span,
    substr: Substructure<'_>,
    discr_then_data: bool,
) -> BlockOrExpr {
    // Builds:
    //
    // match ::core::cmp::Ord::cmp(&self.x, &other.x) {
    //     ::std::cmp::Ordering::Equal =>
    //         ::core::cmp::Ord::cmp(&self.y, &other.y),
    //     cmp => cmp,
    // }
    let expr = cmp_body(cx, span, substr, discr_then_data, OrdlikeDerive::Ord);
    BlockOrExpr::new_expr(expr)
}
