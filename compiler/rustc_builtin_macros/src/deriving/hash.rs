use rustc_ast::{Mutability, Safety};
use rustc_expand::base::ExtCtxt;
use rustc_span::{Ident, Span, kw, sym};
use thin_vec::{ThinVec, thin_vec};

use crate::deriving::generic::*;
use crate::deriving::{call_discriminant_value, path_std};

pub(crate) fn expand_deriving_hash(
    cx: &ExtCtxt<'_>,
    span: Span,
    item: &ast::Item,
    push: &mut dyn FnMut(Box<ast::Item>),
    is_const: bool,
) {
    let path = path_std!(cx, span, hash::Hash);

    let typaram = Ident::new(sym::__H, span);

    let arg = cx.ty_path(cx.path_ident(span, typaram));

    let param = {
        let path = path_std!(cx, span, hash::Hasher);
        cx.typaram(span, typaram, thin_vec![cx.trait_bound(path, false)], None)
    };

    let generics = ast::Generics {
        params: thin_vec![param],
        where_clause: ast::WhereClause { has_where_token: false, predicates: ThinVec::new(), span },
        span,
    };

    let hash_trait_def = TraitDef {
        span,
        path,
        skip_path_as_bound: false,
        needs_copy_as_bound_if_packed: true,
        additional_bounds: SmallVec::new(),
        supports_unions: false,
        methods: smallvec![MethodDef {
            name: sym::hash,
            generics,
            explicit_self: true,
            nonself_args: smallvec![(cx.ty_ref(span, arg, None, Mutability::Mut), sym::state)],
            has_other_selflike_arg: false,
            ret_ty: cx.ty_unit(span),
            attributes: thin_vec![cx.attr_word(sym::inline, span)],
            fieldless_variants_strategy: FieldlessVariantsStrategy::Unify,
            combine_substructure: combine_substructure(hash_substructure),
        }],
        is_const,
        safety: Safety::Default,
        document: true,
    };

    hash_trait_def.expand(cx, item, push);
}

fn hash_substructure(cx: &ExtCtxt<'_>, span: Span, substr: Substructure<'_>) -> BlockOrExpr {
    let call_hash = |span, expr| {
        let strs = cx.std_path(&[sym::hash, sym::Hash, sym::hash]);
        let hash_path = cx.expr_path(cx.path_global(span, strs));
        let expr =
            cx.expr_call(span, hash_path, thin_vec![expr, cx.expr_ident_sym(span, sym::state)]);
        cx.stmt_expr(expr)
    };

    let (stmts, match_expr) = match substr {
        Struct(_, fields) | EnumMatching(.., fields) => {
            let stmts =
                fields.into_iter().map(|field| call_hash(field.span, field.self_expr)).collect();
            (stmts, None)
        }
        EnumDiscr(match_expr) => {
            let stmts = thin_vec![call_hash(
                span,
                cx.expr_addr_of(span, call_discriminant_value(cx, span, kw::SelfLower))
            )];
            (stmts, match_expr)
        }
        _ => cx.dcx().span_bug(span, "unexpected substructure in `derive(Hash)`"),
    };

    BlockOrExpr::new_mixed(stmts, match_expr)
}
