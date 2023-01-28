use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::{path_std, pathvec_std};
use rustc_ast::{ExprKind, ItemKind, MetaItem, PatKind};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{sym, Ident};
use rustc_span::Span;
use thin_vec::thin_vec;

pub fn expand_deriving_partial_ord(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    is_const: bool,
) {
    let ordering_ty = Path(path_std!(cmp::Ordering));
    let ret_ty =
        Path(Path::new_(pathvec_std!(option::Option), vec![Box::new(ordering_ty)], PathKind::Std));

    let attrs = thin_vec![cx.attr_word(sym::inline, span)];

    // Order in which to perform matching
    let tag_then_data = if let Annotatable::Item(item) = item
        && let ItemKind::Enum(def, _) = &item.kind {
            let dataful: Vec<bool> = def.variants.iter().map(|v| !v.data.fields().is_empty()).collect();
            match dataful.iter().filter(|&&b| b).count() {
                // No data, placing the tag check first makes codegen simpler
                0 => true,
                1..=2 => false,
                _ => {
                    (0..dataful.len()-1).any(|i| {
                        if dataful[i] && let Some(idx) = dataful[i+1..].iter().position(|v| *v) {
                            idx >= 2
                        } else {
                            false
                        }
                    })
                }
            }
        } else {
            true
        };
    let partial_cmp_def = MethodDef {
        name: sym::partial_cmp,
        generics: Bounds::empty(),
        explicit_self: true,
        nonself_args: vec![(self_ref(), sym::other)],
        ret_ty,
        attributes: attrs,
        fieldless_variants_strategy: FieldlessVariantsStrategy::Unify,
        combine_substructure: combine_substructure(Box::new(|cx, span, substr| {
            cs_partial_cmp(cx, span, substr, tag_then_data)
        })),
    };

    let trait_def = TraitDef {
        span,
        path: path_std!(cmp::PartialOrd),
        skip_path_as_bound: false,
        additional_bounds: vec![],
        supports_unions: false,
        methods: vec![partial_cmp_def],
        associated_types: Vec::new(),
        is_const,
    };
    trait_def.expand(cx, mitem, item, push)
}

fn cs_partial_cmp(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    substr: &Substructure<'_>,
    tag_then_data: bool,
) -> BlockOrExpr {
    let test_id = Ident::new(sym::cmp, span);
    let equal_path = cx.path_global(span, cx.std_path(&[sym::cmp, sym::Ordering, sym::Equal]));
    let partial_cmp_path = cx.std_path(&[sym::cmp, sym::PartialOrd, sym::partial_cmp]);

    // Builds:
    //
    // match ::core::cmp::PartialOrd::partial_cmp(&self.x, &other.x) {
    //     ::core::option::Option::Some(::core::cmp::Ordering::Equal) =>
    //         ::core::cmp::PartialOrd::partial_cmp(&self.y, &other.y),
    //     cmp => cmp,
    // }
    let expr = cs_fold(
        // foldr nests the if-elses correctly, leaving the first field
        // as the outermost one, and the last as the innermost.
        false,
        cx,
        span,
        substr,
        |cx, fold| match fold {
            CsFold::Single(field) => {
                let [other_expr] = &field.other_selflike_exprs[..] else {
                        cx.span_bug(field.span, "not exactly 2 arguments in `derive(Ord)`");
                    };
                let args = vec![field.self_expr.clone(), other_expr.clone()];
                cx.expr_call_global(field.span, partial_cmp_path.clone(), args)
            }
            CsFold::Combine(span, mut expr1, expr2) => {
                // When the item is an enum, this expands to
                // ```
                // match (expr2) {
                //     Some(Ordering::Equal) => expr1,
                //     cmp => cmp
                // }
                // ```
                // where `expr2` is `partial_cmp(self_tag, other_tag)`, and `expr1` is a `match`
                //  against the enum variants. This means that we begin by comparing the enum tags,
                // before either inspecting their contents (if they match), or returning
                // the `cmp::Ordering` of comparing the enum tags.
                // ```
                // match partial_cmp(self_tag, other_tag) {
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
                //     _ => partial_cmp(self_tag, other_tag)
                // }
                // ```
                // Reference: https://github.com/rust-lang/rust/pull/103659#issuecomment-1328126354

                if !tag_then_data
                    && let ExprKind::Match(_, arms) = &mut expr1.kind
                    && let Some(last) = arms.last_mut()
                    && let PatKind::Wild = last.pat.kind {
                        last.body = expr2;
                        expr1
                } else {
                    let eq_arm =
                        cx.arm(span, cx.pat_some(span, cx.pat_path(span, equal_path.clone())), expr1);
                    let neq_arm =
                        cx.arm(span, cx.pat_ident(span, test_id), cx.expr_ident(span, test_id));
                    cx.expr_match(span, expr2, vec![eq_arm, neq_arm])
                }
            }
            CsFold::Fieldless => cx.expr_some(span, cx.expr_path(equal_path.clone())),
        },
    );
    BlockOrExpr::new_expr(expr)
}
