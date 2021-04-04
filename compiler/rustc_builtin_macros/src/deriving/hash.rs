use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::{self, path_std, pathvec_std};

use rustc_ast::ptr::P;
use rustc_ast::{Expr, MetaItem, Mutability};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::sym;
use rustc_span::Span;

pub fn expand_deriving_hash(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
) {
    let path = Path::new_(pathvec_std!(hash::Hash), None, vec![], PathKind::Std);

    let typaram = sym::__H;

    let arg = Path::new_local(typaram);
    let hash_trait_def = TraitDef {
        span,
        attributes: Vec::new(),
        path,
        additional_bounds: Vec::new(),
        generics: Bounds::empty(),
        is_unsafe: false,
        supports_unions: false,
        methods: vec![MethodDef {
            name: sym::hash,
            generics: Bounds { bounds: vec![(typaram, vec![path_std!(hash::Hasher)])] },
            explicit_self: borrowed_explicit_self(),
            args: vec![(Ptr(Box::new(Literal(arg)), Borrowed(None, Mutability::Mut)), sym::state)],
            ret_ty: nil_ty(),
            attributes: vec![],
            is_unsafe: false,
            unify_fieldless_variants: true,
            combine_substructure: combine_substructure(Box::new(|a, b, c| {
                hash_substructure(a, b, c)
            })),
        }],
        associated_types: Vec::new(),
    };

    hash_trait_def.expand(cx, mitem, item, push);
}

fn hash_substructure(cx: &mut ExtCtxt<'_>, trait_span: Span, substr: &Substructure<'_>) -> P<Expr> {
    let state_expr = match substr.nonself_args {
        [o_f] => o_f,
        _ => cx.span_bug(trait_span, "incorrect number of arguments in `derive(Hash)`"),
    };
    let call_hash = |span, thing_expr| {
        let hash_path = {
            let strs = cx.std_path(&[sym::hash, sym::Hash, sym::hash]);

            cx.expr_path(cx.path_global(span, strs))
        };
        let ref_thing = cx.expr_addr_of(span, thing_expr);
        let expr = cx.expr_call(span, hash_path, vec![ref_thing, state_expr.clone()]);
        cx.stmt_expr(expr)
    };
    let mut stmts = Vec::new();

    let fields = match substr.fields {
        Struct(_, fs) | EnumMatching(_, 1, .., fs) => fs,
        EnumMatching(.., fs) => {
            let variant_value = deriving::call_intrinsic(
                cx,
                trait_span,
                sym::discriminant_value,
                vec![cx.expr_self(trait_span)],
            );

            stmts.push(call_hash(trait_span, variant_value));

            fs
        }
        _ => cx.span_bug(trait_span, "impossible substructure in `derive(Hash)`"),
    };

    stmts.extend(
        fields.iter().map(|FieldInfo { ref self_, span, .. }| call_hash(*span, self_.clone())),
    );

    cx.expr_block(cx.block(trait_span, stmts))
}
