use rustc_ast::{BinOpKind, BorrowKind, Expr, ExprKind, MetaItem, Mutability};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::{Span, sym};
use thin_vec::thin_vec;

use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::{path_local, path_std};

/// Expands a `#[derive(PartialEq)]` attribute into an implementation for the
/// target item.
pub(crate) fn expand_deriving_partial_eq(
    cx: &ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    is_const: bool,
) {
    let structural_trait_def = TraitDef {
        span,
        path: path_std!(marker::StructuralPartialEq),
        skip_path_as_bound: true, // crucial!
        needs_copy_as_bound_if_packed: false,
        additional_bounds: Vec::new(),
        // We really don't support unions, but that's already checked by the impl generated below;
        // a second check here would lead to redundant error messages.
        supports_unions: true,
        methods: Vec::new(),
        associated_types: Vec::new(),
        is_const: false,
        is_staged_api_crate: cx.ecfg.features.staged_api(),
    };
    structural_trait_def.expand(cx, mitem, item, push);

    // No need to generate `ne`, the default suffices, and not generating it is
    // faster.
    let methods = vec![MethodDef {
        name: sym::eq,
        generics: Bounds::empty(),
        explicit_self: true,
        nonself_args: vec![(self_ref(), sym::other)],
        ret_ty: Path(path_local!(bool)),
        attributes: thin_vec![cx.attr_word(sym::inline, span)],
        fieldless_variants_strategy: FieldlessVariantsStrategy::Unify,
        combine_substructure: combine_substructure(Box::new(|a, b, c| {
            BlockOrExpr::new_expr(get_substructure_equality_expr(a, b, c))
        })),
    }];

    let trait_def = TraitDef {
        span,
        path: path_std!(cmp::PartialEq),
        skip_path_as_bound: false,
        needs_copy_as_bound_if_packed: true,
        additional_bounds: Vec::new(),
        supports_unions: false,
        methods,
        associated_types: Vec::new(),
        is_const,
        is_staged_api_crate: cx.ecfg.features.staged_api(),
    };
    trait_def.expand(cx, mitem, item, push)
}

/// Generates the equality expression for a struct or enum variant when deriving
/// `PartialEq`.
///
/// This function generates an expression that checks if all fields of a struct
/// or enum variant are equal.
/// - Scalar fields are compared first for efficiency, followed by compound
///   fields.
/// - If there are no fields, returns `true` (fieldless types are always equal).
///
/// Whether a field is considered "scalar" is determined by comparing the symbol
/// of its type to a set of known scalar type symbols (e.g., `i32`, `u8`, etc).
/// This check is based on the type's symbol.
///
/// ### Example 1
/// ```
/// #[derive(PartialEq)]
/// struct i32;
///
/// // Here, `field_2` is of type `i32`, but since it's a user-defined type (not
/// // the primitive), it will not be treated as scalar. The function will still
/// // check equality of `field_2` first because the symbol matches `i32`.
/// #[derive(PartialEq)]
/// struct Struct {
///     field_1: &'static str,
///     field_2: i32,
/// }
/// ```
///
/// ### Example 2
/// ```
/// mod ty {
///     pub type i32 = i32;
/// }
///
/// // Here, `field_2` is of type `ty::i32`, which is a type alias for `i32`.
/// // However, the function will not reorder the fields because the symbol for
/// // `ty::i32` does not match the symbol for the primitive `i32`
/// // ("ty::i32" != "i32").
/// #[derive(PartialEq)]
/// struct Struct {
///     field_1: &'static str,
///     field_2: ty::i32,
/// }
/// ```
///
/// For enums, the discriminant is compared first, then the rest of the fields.
///
/// # Panics
///
/// If called on static or all-fieldless enums/structs, which should not occur
/// during derive expansion.
fn get_substructure_equality_expr(
    cx: &ExtCtxt<'_>,
    span: Span,
    substructure: &Substructure<'_>,
) -> Box<Expr> {
    use SubstructureFields::*;

    match substructure.fields {
        EnumMatching(.., fields) | Struct(.., fields) => {
            let combine = move |acc, field| {
                let rhs = get_field_equality_expr(cx, field);
                if let Some(lhs) = acc {
                    // Combine the previous comparison with the current field
                    // using logical AND.
                    return Some(cx.expr_binary(field.span, BinOpKind::And, lhs, rhs));
                }
                // Start the chain with the first field's comparison.
                Some(rhs)
            };

            // First compare scalar fields, then compound fields, combining all
            // with logical AND.
            return fields
                .iter()
                .filter(|field| !field.maybe_scalar)
                .fold(fields.iter().filter(|field| field.maybe_scalar).fold(None, combine), combine)
                // If there are no fields, treat as always equal.
                .unwrap_or_else(|| cx.expr_bool(span, true));
        }
        EnumDiscr(disc, match_expr) => {
            let lhs = get_field_equality_expr(cx, disc);
            let Some(match_expr) = match_expr else {
                return lhs;
            };
            // Compare the discriminant first (cheaper), then the rest of the
            // fields.
            return cx.expr_binary(disc.span, BinOpKind::And, lhs, match_expr.clone());
        }
        StaticEnum(..) => cx.dcx().span_bug(
            span,
            "unexpected static enum encountered during `derive(PartialEq)` expansion",
        ),
        StaticStruct(..) => cx.dcx().span_bug(
            span,
            "unexpected static struct encountered during `derive(PartialEq)` expansion",
        ),
        AllFieldlessEnum(..) => cx.dcx().span_bug(
            span,
            "unexpected all-fieldless enum encountered during `derive(PartialEq)` expansion",
        ),
    }
}

/// Generates an equality comparison expression for a single struct or enum
/// field.
///
/// This function produces an AST expression that compares the `self` and
/// `other` values for a field using `==`. It removes any leading references
/// from both sides for readability. If the field is a block expression, it is
/// wrapped in parentheses to ensure valid syntax.
///
/// # Panics
///
/// Panics if there are not exactly two arguments to compare (should be `self`
/// and `other`).
fn get_field_equality_expr(cx: &ExtCtxt<'_>, field: &FieldInfo) -> Box<Expr> {
    let [rhs] = &field.other_selflike_exprs[..] else {
        cx.dcx().span_bug(field.span, "not exactly 2 arguments in `derive(PartialEq)`");
    };

    cx.expr_binary(
        field.span,
        BinOpKind::Eq,
        wrap_block_expr(cx, peel_refs(&field.self_expr)),
        wrap_block_expr(cx, peel_refs(rhs)),
    )
}

/// Removes all leading immutable references from an expression.
///
/// This is used to strip away any number of leading `&` from an expression
/// (e.g., `&&&T` becomes `T`). Only removes immutable references; mutable
/// references are preserved.
fn peel_refs(mut expr: &Box<Expr>) -> Box<Expr> {
    while let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, inner) = &expr.kind {
        expr = &inner;
    }
    expr.clone()
}

/// Wraps a block expression in parentheses to ensure valid AST in macro
/// expansion output.
///
/// If the given expression is a block, it is wrapped in parentheses; otherwise,
/// it is returned unchanged.
fn wrap_block_expr(cx: &ExtCtxt<'_>, expr: Box<Expr>) -> Box<Expr> {
    if matches!(&expr.kind, ExprKind::Block(..)) {
        return cx.expr_paren(expr.span, expr);
    }
    expr
}
