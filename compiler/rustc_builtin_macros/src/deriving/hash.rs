use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::{path_std, pathvec_std};
use rustc_ast::{AttrVec, MetaItem, Mutability};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::sym;
use rustc_span::Span;
use thin_vec::thin_vec;

pub fn expand_deriving_hash(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    is_const: bool,
) {
    let path = Path::new_(pathvec_std!(hash::Hash), vec![], PathKind::Std);

    let typaram = sym::__H;

    let arg = Path::new_local(typaram);
    let hash_trait_def = TraitDef {
        span,
        path,
        skip_path_as_bound: false,
        needs_copy_as_bound_if_packed: true,
        additional_bounds: Vec::new(),
        supports_unions: false,
        methods: vec![MethodDef {
            name: sym::hash,
            generics: Bounds { bounds: vec![(typaram, vec![path_std!(hash::Hasher)])] },
            explicit_self: true,
            nonself_args: vec![(Ref(Box::new(Path(arg)), Mutability::Mut), sym::state)],
            ret_ty: Unit,
            attributes: AttrVec::new(),
            fieldless_variants_strategy: FieldlessVariantsStrategy::Unify,
            combine_substructure: combine_substructure(Box::new(|a, b, c| {
                hash_substructure(a, b, c)
            })),
        }],
        associated_types: Vec::new(),
        is_const,
    };

    hash_trait_def.expand(cx, mitem, item, push);
}

fn hash_substructure(
    cx: &mut ExtCtxt<'_>,
    trait_span: Span,
    substr: &Substructure<'_>,
) -> BlockOrExpr {
    let [state_expr] = substr.nonselflike_args else {
        cx.span_bug(trait_span, "incorrect number of arguments in `derive(Hash)`");
    };
    let call_hash = |span, expr| {
        let hash_path = {
            let strs = cx.std_path(&[sym::hash, sym::Hash, sym::hash]);

            cx.expr_path(cx.path_global(span, strs))
        };
        let expr = cx.expr_call(span, hash_path, thin_vec![expr, state_expr.clone()]);
        cx.stmt_expr(expr)
    };

    let (stmts, match_expr) = match substr.fields {
        Struct(_, fields) | EnumMatching(.., fields) => {
            let stmts =
                fields.iter().map(|field| call_hash(field.span, field.self_expr.clone())).collect();
            (stmts, None)
        }
        EnumTag(tag_field, match_expr) => {
            assert!(tag_field.other_selflike_exprs.is_empty());
            let stmts = thin_vec![call_hash(tag_field.span, tag_field.self_expr.clone())];
            (stmts, match_expr.clone())
        }
        _ => cx.span_bug(trait_span, "impossible substructure in `derive(Hash)`"),
    };

    BlockOrExpr::new_mixed(stmts, match_expr)
}
